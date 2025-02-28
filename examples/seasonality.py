"""
Add time-based effects and seasonality to the MMM model.
"""


def add_time_based_effects(mmm):
    """
    Add time-based effects and seasonality indicators to the model.

    Args:
        mmm: MarketingMixModel instance with preprocessed data

    Returns:
        Updated MarketingMixModel with time-based features added
    """
    # Make sure we have preprocessed data
    if mmm.preprocessed_data is None:
        raise ValueError("No preprocessed data available. Call preprocess_data() first.")

    # Make sure we have a date column
    if 'date' not in mmm.preprocessed_data.columns:
        print("No date column found, cannot add time-based effects")
        return mmm

    # Extract date components
    mmm.preprocessed_data['year'] = mmm.preprocessed_data['date'].dt.year
    mmm.preprocessed_data['quarter'] = mmm.preprocessed_data['date'].dt.quarter
    mmm.preprocessed_data['month'] = mmm.preprocessed_data['date'].dt.month
    mmm.preprocessed_data['week_of_year'] = mmm.preprocessed_data['date'].dt.isocalendar().week
    mmm.preprocessed_data['day_of_week'] = mmm.preprocessed_data['date'].dt.dayofweek

    # Add quarter dummies (Q1 is reference)
    mmm.preprocessed_data['quarter_2'] = (mmm.preprocessed_data['quarter'] == 2).astype(int)
    mmm.preprocessed_data['quarter_3'] = (mmm.preprocessed_data['quarter'] == 3).astype(int)
    mmm.preprocessed_data['quarter_4'] = (mmm.preprocessed_data['quarter'] == 4).astype(int)

    # Add month dummies (if data has sufficient granularity)
    month_counts = mmm.preprocessed_data['month'].value_counts()
    if len(month_counts) > 6:  # Only add if we have reasonable month coverage
        for month in range(2, 13):  # January is reference
            mmm.preprocessed_data[f'month_{month}'] = (mmm.preprocessed_data['month'] == month).astype(int)

    # Add time trend
    mmm.preprocessed_data['time_trend'] = np.arange(len(mmm.preprocessed_data))
    mmm.preprocessed_data['time_trend_squared'] = mmm.preprocessed_data['time_trend'] ** 2

    # Add holiday indicators
    add_holiday_indicators(mmm)

    # Determine which seasonal variables to add to features
    # Prioritize quarters for cleaner models, months if more granularity needed
    if len(month_counts) > 6:
        seasonal_features = ['quarter_2', 'quarter_3', 'quarter_4', 'time_trend']
    else:
        seasonal_features = ['quarter_2', 'quarter_3', 'quarter_4', 'time_trend']

    # Add holiday features if they exist
    holiday_features = [col for col in mmm.preprocessed_data.columns if 'holiday_' in col]
    seasonal_features.extend(holiday_features)

    # Add features to model
    mmm.feature_names.extend(seasonal_features)

    # Update X to include seasonal features
    mmm.X = mmm.preprocessed_data[mmm.feature_names]

    print(f"Added time-based effects: {', '.join(seasonal_features)}")

    return mmm


def add_holiday_indicators(mmm):
    """
    Add holiday period indicators based on common shopping seasons.
    """
    # Initialize holiday columns
    mmm.preprocessed_data['holiday_blackfriday'] = 0
    mmm.preprocessed_data['holiday_christmas'] = 0
    mmm.preprocessed_data['holiday_newyear'] = 0
    mmm.preprocessed_data['holiday_summer'] = 0

    # For each date, check if it falls in a holiday period
    for idx, row in mmm.preprocessed_data.iterrows():
        date = row['date']
        month = date.month
        day = date.day

        # Black Friday / Thanksgiving (late November)
        if month == 11 and day >= 20:
            mmm.preprocessed_data.at[idx, 'holiday_blackfriday'] = 1

        # Christmas period (December)
        if month == 12 and day >= 1:
            mmm.preprocessed_data.at[idx, 'holiday_christmas'] = 1

        # New Year period (early January)
        if month == 1 and day <= 15:
            mmm.preprocessed_data.at[idx, 'holiday_newyear'] = 1

        # Summer period (July-August)
        if month in [7, 8]:
            mmm.preprocessed_data.at[idx, 'holiday_summer'] = 1


def evaluate_seasonality_impact(data, target='revenue', media_cols=None, control_cols=None):
    """
    Evaluate the impact of adding seasonality on model performance.

    Args:
        data: DataFrame with media data
        target: Target variable name
        media_cols: Media spending columns
        control_cols: Control variables

    Returns:
        Dictionary with model results
    """
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

    # 1. Baseline model without seasonality
    print("\nTraining baseline model without seasonality...")
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

    # 2. Model with seasonality
    print("Training model with seasonality...")
    mmm_seasonal = MarketingMixModel()
    mmm_seasonal.load_data_from_dataframe(train_data)

    mmm_seasonal.preprocess_data(
        target=target,
        date_col='date' if 'date' in train_data.columns else None,
        media_cols=transformed_cols,
        control_cols=control_cols
    )

    mmm_seasonal.apply_adstock_to_all_media(
        media_cols=transformed_cols,
        decay_rates=decay_rates,
        max_lags=max_lags
    )

    # Add seasonality
    mmm_seasonal = add_time_based_effects(mmm_seasonal)

    seasonal_results = mmm_seasonal.fit_model()
    seasonal_r2 = seasonal_results.rsquared
    seasonal_adj_r2 = seasonal_results.rsquared_adj

    # Predict on test set - need to add seasonal features to test data
    test_data_seasonal = test_data.copy()

    # Add the same seasonal features to test data
    seasonal_features = [col for col in mmm_seasonal.feature_names
                         if col not in mmm_baseline.feature_names]

    # Add necessary date components to test data
    for feat in seasonal_features:
        if feat == 'quarter_2':
            test_data_seasonal['quarter'] = test_data_seasonal['date'].dt.quarter
            test_data_seasonal['quarter_2'] = (test_data_seasonal['quarter'] == 2).astype(int)
        elif feat == 'quarter_3':
            if 'quarter' not in test_data_seasonal.columns:
                test_data_seasonal['quarter'] = test_data_seasonal['date'].dt.quarter
            test_data_seasonal['quarter_3'] = (test_data_seasonal['quarter'] == 3).astype(int)
        elif feat == 'quarter_4':
            if 'quarter' not in test_data_seasonal.columns:
                test_data_seasonal['quarter'] = test_data_seasonal['date'].dt.quarter
            test_data_seasonal['quarter_4'] = (test_data_seasonal['quarter'] == 4).astype(int)
        elif feat.startswith('month_'):
            month_num = int(feat.split('_')[1])
            test_data_seasonal['month'] = test_data_seasonal['date'].dt.month
            test_data_seasonal[feat] = (test_data_seasonal['month'] == month_num).astype(int)
        elif feat == 'time_trend':
            # Continue the trend from training data
            start_trend = len(train_data)
            test_data_seasonal['time_trend'] = np.arange(start_trend, start_trend + len(test_data))
        elif feat == 'time_trend_squared':
            if 'time_trend' not in test_data_seasonal.columns:
                start_trend = len(train_data)
                test_data_seasonal['time_trend'] = np.arange(start_trend, start_trend + len(test_data))
            test_data_seasonal['time_trend_squared'] = test_data_seasonal['time_trend'] ** 2
        elif feat.startswith('holiday_'):
            # Re-create holiday indicators for test data
            test_data_seasonal[feat] = 0
            for idx, row in test_data_seasonal.iterrows():
                date = row['date']
                month = date.month
                day = date.day

                if feat == 'holiday_blackfriday' and month == 11 and day >= 20:
                    test_data_seasonal.at[idx, feat] = 1
                elif feat == 'holiday_christmas' and month == 12:
                    test_data_seasonal.at[idx, feat] = 1
                elif feat == 'holiday_newyear' and month == 1 and day <= 15:
                    test_data_seasonal.at[idx, feat] = 1
                elif feat == 'holiday_summer' and month in [7, 8]:
                    test_data_seasonal.at[idx, feat] = 1

    X_test_seasonal = sm.add_constant(test_data_seasonal[mmm_seasonal.feature_names])
    seasonal_preds = seasonal_results.predict(X_test_seasonal)

    seasonal_test_r2 = r2_score(y_test, seasonal_preds)
    seasonal_test_rmse = np.sqrt(mean_squared_error(y_test, seasonal_preds))
    seasonal_test_mape = mean_absolute_percentage_error(y_test, seasonal_preds) * 100

    # Compare seasonality coefficients
    seasonal_terms = [col for col in mmm_seasonal.feature_names
                      if col not in mmm_baseline.feature_names]
    significant_seasonal = []

    if seasonal_terms:
        print("\nSeasonality Effects:")
        for term in seasonal_terms:
            if term in seasonal_results.params:
                coef = seasonal_results.params[term]
                pval = seasonal_results.pvalues[term]
                print(f"  {term}: coefficient={coef:.4f}, p-value={pval:.4f}")

                if pval < 0.1:  # Using 0.1 as threshold for significance
                    significant_seasonal.append((term, coef, pval))

    # Create comparison table
    print("\nModel Performance Comparison:")
    print(f"                        | Without Seasonality | With Seasonality | Difference")
    print(f"------------------------|--------------------|-------------------|----------")
    print(
        f"Training R²             | {baseline_r2:.4f}            | {seasonal_r2:.4f}           | {seasonal_r2 - baseline_r2:.4f}")
    print(
        f"Training Adj. R²        | {baseline_adj_r2:.4f}            | {seasonal_adj_r2:.4f}           | {seasonal_adj_r2 - baseline_adj_r2:.4f}")
    print(
        f"Test R²                 | {baseline_test_r2:.4f}            | {seasonal_test_r2:.4f}           | {seasonal_test_r2 - baseline_test_r2:.4f}")
    print(
        f"Test RMSE               | {baseline_test_rmse:.2f}          | {seasonal_test_rmse:.2f}         | {seasonal_test_rmse - baseline_test_rmse:.2f}")
    print(
        f"Test MAPE               | {baseline_test_mape:.2f}%          | {seasonal_test_mape:.2f}%         | {seasonal_test_mape - baseline_test_mape:.2f}%")

    if significant_seasonal:
        print("\nSignificant Seasonal Effects (p < 0.1):")
        for term, coef, pval in significant_seasonal:
            print(f"  {term}: coefficient={coef:.4f}, p-value={pval:.4f}")
    else:
        print("\nNo statistically significant seasonal effects found.")

    # Plot seasonality patterns
    if 'quarter' in mmm_seasonal.preprocessed_data.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='quarter', y=target, data=mmm_seasonal.preprocessed_data)
        plt.title(f'{target} by Quarter')
        plt.tight_layout()
        plt.savefig('seasonality_by_quarter.png')

    if 'month' in mmm_seasonal.preprocessed_data.columns:
        plt.figure(figsize=(14, 6))
        sns.boxplot(x='month', y=target, data=mmm_seasonal.preprocessed_data)
        plt.title(f'{target} by Month')
        plt.tight_layout()
        plt.savefig('seasonality_by_month.png')

    # Plot actual vs predicted with seasonality
    plt.figure(figsize=(14, 7))

    # Plot training data
    train_dates = train_data['date']
    train_actual = train_data[target]
    train_preds = seasonal_results.predict(sm.add_constant(train_data[mmm_seasonal.feature_names]))

    # Plot test data
    test_dates = test_data['date']

    plt.plot(train_dates, train_actual, 'b-', alpha=0.6, label='Training Actual')
    plt.plot(train_dates, train_preds, 'g--', alpha=0.6, label='Training Predicted')
    plt.plot(test_dates, y_test, 'b-', label='Test Actual')
    plt.plot(test_dates, seasonal_preds, 'r--', label='Test Predicted (with Seasonality)')

    plt.title('Revenue with Seasonal Effects')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('seasonal_model_performance.png')

    # Return results
    return {
        'baseline': {
            'model': baseline_results,
            'train_r2': baseline_r2,
            'test_r2': baseline_test_r2,
            'test_rmse': baseline_test_rmse,
            'test_mape': baseline_test_mape
        },
        'seasonal': {
            'model': seasonal_results,
            'train_r2': seasonal_r2,
            'test_r2': seasonal_test_r2,
            'test_rmse': seasonal_test_rmse,
            'test_mape': seasonal_test_mape,
            'significant_seasonal': significant_seasonal
        }
    }


# You can run this as part of your test script or as a standalone analysis
if __name__ == "__main__":
    # Load data
    data_path = Path(__file__).parent / 'mock_marketing_data.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])

    # Evaluate seasonality impact
    seasonality_results = evaluate_seasonality_impact(data)

    print("Seasonality analysis complete.")