"""
Add channel interaction effects to capture synergies between marketing channels.
"""


def add_channel_interactions(mmm, channels=None):
    """
    Add interaction terms between selected media channels to capture synergies.

    Args:
        mmm: MarketingMixModel instance with preprocessed data
        channels: List of channel pairs to create interactions for. If None, use predefined pairs.

    Returns:
        Updated MarketingMixModel with interaction terms added
    """
    # Make sure we have preprocessed data
    if mmm.preprocessed_data is None:
        raise ValueError("No preprocessed data available. Call preprocess_data() first.")

    # Default channel interactions to test
    if channels is None:
        # These are common synergistic pairs in marketing
        channels = [
            ('tv_spend_log_adstocked', 'search_spend_log_adstocked'),  # TV drives search
            ('tv_spend_log_adstocked', 'social_media_spend_log_adstocked'),  # TV amplifies social
            ('digital_display_spend_log_adstocked', 'search_spend_log_adstocked'),  # Display + Search
            ('social_media_spend_log_adstocked', 'search_spend_log_adstocked')  # Social + Search
        ]

    # Add interaction terms
    for channel1, channel2 in channels:
        # Check if both channels exist
        if channel1 in mmm.preprocessed_data.columns and channel2 in mmm.preprocessed_data.columns:
            # Create interaction term
            interaction_name = f"{channel1.split('_')[0]}_{channel2.split('_')[0]}_interaction"
            mmm.preprocessed_data[interaction_name] = mmm.preprocessed_data[channel1] * mmm.preprocessed_data[channel2]

            # Add to feature names
            mmm.feature_names.append(interaction_name)

            print(f"Added interaction term: {interaction_name}")
        else:
            print(f"Couldn't add interaction: {channel1} or {channel2} not found")

    # Update X to include interaction terms
    mmm.X = mmm.preprocessed_data[mmm.feature_names]

    return mmm


def evaluate_interactions_impact(data, target='revenue', media_cols=None, control_cols=None):
    """
    Evaluate the impact of adding channel interactions on model performance.

    Args:
        data: DataFrame with media data
        target: Target variable name
        media_cols: Media spending columns
        control_cols: Control variables

    Returns:
        Dictionary with model results
    """
    from sklearn.model_selection import train_test_split

    if media_cols is None:
        media_cols = [
            'tv_spend',
            'digital_display_spend',
            'search_spend',
            'social_media_spend',
            'video_spend',
            'email_spend'
        ]

    if control_cols is None:
        control_cols = [
            'price_index',
            'competitor_price_index',
            'gdp_index',
            'consumer_confidence'
        ]

    # Create log-transformed variables
    transformed_data = data.copy()
    for col in media_cols:
        transformed_data[f"{col}_log"] = np.log1p(transformed_data[col])

    transformed_cols = [f"{col}_log" for col in media_cols]

    # Setup decay rates and lags
    decay_rates = {
        'tv_spend_log': 0.85,
        'digital_display_spend_log': 0.7,
        'search_spend_log': 0.3,
        'social_media_spend_log': 0.6,
        'video_spend_log': 0.75,
        'email_spend_log': 0.4
    }

    max_lags = {
        'tv_spend_log': 8,
        'digital_display_spend_log': 4,
        'search_spend_log': 2,
        'social_media_spend_log': 5,
        'video_spend_log': 6,
        'email_spend_log': 3
    }

    # Split data chronologically
    if 'date' in transformed_data.columns:
        transformed_data = transformed_data.sort_values('date')

    train_size = int(len(transformed_data) * 0.8)
    train_data = transformed_data.iloc[:train_size]
    test_data = transformed_data.iloc[train_size:]

    # 1. Baseline model without interactions
    print("\nTraining baseline model without interactions...")
    mmm_baseline = MarketingMixModel()
    mmm_baseline.load_data_from_dataframe(train_data)

    mmm_baseline.preprocess_data(
        target=target,
        date_col='date' if 'date' in train_data.columns else None,
        media_cols=transformed_cols,
        control_cols=control_cols
    )

    mmm_baseline.apply_adstock_to_all_media(
        media_cols=transformed_cols,
        decay_rates=decay_rates,
        max_lags=max_lags
    )

    baseline_results = mmm_baseline.fit_model()
    baseline_r2 = baseline_results.rsquared
    baseline_adj_r2 = baseline_results.rsquared_adj

    # Predict on test set
    import statsmodels.api as sm
    X_test = sm.add_constant(test_data[mmm_baseline.feature_names])
    y_test = test_data[target]
    baseline_preds = baseline_results.predict(X_test)

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
    baseline_test_r2 = r2_score(y_test, baseline_preds)
    baseline_test_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    baseline_test_mape = mean_absolute_percentage_error(y_test, baseline_preds) * 100

    # 2. Model with interactions
    print("Training model with channel interactions...")
    mmm_interact = MarketingMixModel()
    mmm_interact.load_data_from_dataframe(train_data)

    mmm_interact.preprocess_data(
        target=target,
        date_col='date' if 'date' in train_data.columns else None,
        media_cols=transformed_cols,
        control_cols=control_cols
    )

    mmm_interact.apply_adstock_to_all_media(
        media_cols=transformed_cols,
        decay_rates=decay_rates,
        max_lags=max_lags
    )

    # Add interactions
    mmm_interact = add_channel_interactions(mmm_interact)

    interact_results = mmm_interact.fit_model()
    interact_r2 = interact_results.rsquared
    interact_adj_r2 = interact_results.rsquared_adj

    # Predict on test set
    X_test_interact = sm.add_constant(test_data[mmm_interact.feature_names])
    # Handle missing interaction columns in test data
    for col in mmm_interact.feature_names:
        if col not in test_data.columns:
            # Create interaction in test data
            if "interaction" in col:
                parts = col.split('_')
                if len(parts) >= 3:
                    channel1 = f"{parts[0]}_spend_log_adstocked"
                    channel2 = f"{parts[1]}_spend_log_adstocked"
                    if channel1 in test_data.columns and channel2 in test_data.columns:
                        test_data[col] = test_data[channel1] * test_data[channel2]

    X_test_interact = sm.add_constant(test_data[mmm_interact.feature_names])
    interact_preds = interact_results.predict(X_test_interact)

    interact_test_r2 = r2_score(y_test, interact_preds)
    interact_test_rmse = np.sqrt(mean_squared_error(y_test, interact_preds))
    interact_test_mape = mean_absolute_percentage_error(y_test, interact_preds) * 100

    # Compare interaction coefficients
    interaction_terms = [col for col in mmm_interact.feature_names if 'interaction' in col]
    significant_interactions = []

    if interaction_terms:
        print("\nInteraction Effects:")
        for term in interaction_terms:
            if term in interact_results.params:
                coef = interact_results.params[term]
                pval = interact_results.pvalues[term]
                print(f"  {term}: coefficient={coef:.4f}, p-value={pval:.4f}")

                if pval < 0.1:  # Using 0.1 as threshold for interaction significance
                    significant_interactions.append((term, coef, pval))

    # Create comparison table
    print("\nModel Performance Comparison:")
    print(f"                        | Without Interactions | With Interactions | Difference")
    print(f"------------------------|--------------------|-------------------|----------")
    print(
        f"Training R²             | {baseline_r2:.4f}            | {interact_r2:.4f}           | {interact_r2 - baseline_r2:.4f}")
    print(
        f"Training Adj. R²        | {baseline_adj_r2:.4f}            | {interact_adj_r2:.4f}           | {interact_adj_r2 - baseline_adj_r2:.4f}")
    print(
        f"Test R²                 | {baseline_test_r2:.4f}            | {interact_test_r2:.4f}           | {interact_test_r2 - baseline_test_r2:.4f}")
    print(
        f"Test RMSE               | {baseline_test_rmse:.2f}          | {interact_test_rmse:.2f}         | {interact_test_rmse - baseline_test_rmse:.2f}")
    print(
        f"Test MAPE               | {baseline_test_mape:.2f}%          | {interact_test_mape:.2f}%         | {interact_test_mape - baseline_test_mape:.2f}%")

    if significant_interactions:
        print("\nSignificant Interaction Effects (p < 0.1):")
        for term, coef, pval in significant_interactions:
            print(f"  {term}: coefficient={coef:.4f}, p-value={pval:.4f}")
    else:
        print("\nNo statistically significant interaction effects found.")

    # Return results
    return {
        'baseline': {
            'model': baseline_results,
            'train_r2': baseline_r2,
            'test_r2': baseline_test_r2,
            'test_rmse': baseline_test_rmse,
            'test_mape': baseline_test_mape
        },
        'interactions': {
            'model': interact_results,
            'train_r2': interact_r2,
            'test_r2': interact_test_r2,
            'test_rmse': interact_test_rmse,
            'test_mape': interact_test_mape,
            'significant_interactions': significant_interactions
        }
    }


# You can run this as part of your test script or as a standalone analysis
if __name__ == "__main__":
    # Load data
    data_path = Path(__file__).parent / 'mock_marketing_data.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])

    # Evaluate interactions
    interaction_results = evaluate_interactions_impact(data)

    print("Channel interaction analysis complete.")
