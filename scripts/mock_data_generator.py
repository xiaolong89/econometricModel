import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os


def generate_mock_mmm_data(start_date='2020-01-01', end_date='2023-12-31',
                           frequency='W-SUN', random_seed=42):
    """
    Generate synthetic marketing mix modeling data with realistic patterns.

    Args:
        start_date: Start date for the data
        end_date: End date for the data
        frequency: Pandas frequency string (W-SUN for weekly Sunday-ending data)
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame containing synthetic marketing data
    """
    np.random.seed(random_seed)

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    df = pd.DataFrame({'date': dates})

    # Add time dimensions
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Create holiday flags (simplified approach)
    holiday_weeks = []
    for year in df['year'].unique():
        # Thanksgiving week (end of November)
        thanksgiving = pd.Timestamp(f'{year}-11-26') - pd.Timedelta(days=pd.Timestamp(f'{year}-11-26').dayofweek)
        # Christmas week
        christmas = pd.Timestamp(f'{year}-12-25') - pd.Timedelta(days=pd.Timestamp(f'{year}-12-25').dayofweek)
        # New Year week
        new_year = pd.Timestamp(f'{year}-01-01') - pd.Timedelta(days=pd.Timestamp(f'{year}-01-01').dayofweek)
        # July 4th week
        july_fourth = pd.Timestamp(f'{year}-07-04') - pd.Timedelta(days=pd.Timestamp(f'{year}-07-04').dayofweek)

        holiday_weeks.extend([thanksgiving, christmas, new_year, july_fourth])

    df['is_holiday_period'] = df['date'].isin(holiday_weeks)

    # Generate base marketing spend patterns with realistic characteristics

    # 1. Base level with increasing trend
    num_periods = len(df)

    # Base functions to reuse
    def add_trend(periods, slope=0.1):
        return np.linspace(0, slope * periods, periods)

    def add_seasonality(periods, amplitude=1, cycles=1):
        x = np.linspace(0, 2 * np.pi * cycles, periods)
        return amplitude * np.sin(x)

    def add_noise(periods, scale=0.1):
        return np.random.normal(0, scale, periods)

    # Create base marketing spend patterns with realistic variations

    # TV spend: higher in Q4, lower in summer
    tv_base = 100000 + add_trend(num_periods, 100) + 30000 * add_seasonality(num_periods, cycles=1)
    tv_base = tv_base * (1 + 0.2 * df['quarter'].apply(lambda q: 1 if q == 4 else (-0.2 if q == 3 else 0)))
    df['tv_spend'] = np.maximum(tv_base + add_noise(num_periods, 10000), 0)

    # Digital Display: steady increase, less seasonal
    df['digital_display_spend'] = 50000 + add_trend(num_periods, 200) + 5000 * add_seasonality(num_periods,
                                                                                               cycles=3) + add_noise(
        num_periods, 5000)

    # Search: steady with weekly fluctuations
    df['search_spend'] = 40000 + add_trend(num_periods, 50) + 8000 * add_seasonality(num_periods,
                                                                                     cycles=12) + add_noise(num_periods,
                                                                                                            3000)

    # Social: growing faster than other channels
    df['social_media_spend'] = 30000 + add_trend(num_periods, 250) + 10000 * add_seasonality(num_periods,
                                                                                             cycles=2) + add_noise(
        num_periods, 6000)

    # Video: higher spend during key periods
    df['video_spend'] = 45000 + add_trend(num_periods, 150) + 15000 * add_seasonality(num_periods,
                                                                                      cycles=1.5) + add_noise(
        num_periods, 8000)

    # Email: constant with slight growth
    df['email_spend'] = 10000 + add_trend(num_periods, 30) + add_noise(num_periods, 1000)

    # Adjust spend for holiday periods
    holiday_multipliers = {
        'tv_spend': 1.5,
        'digital_display_spend': 1.3,
        'search_spend': 1.4,
        'social_media_spend': 1.25,
        'video_spend': 1.35,
        'email_spend': 1.2
    }

    for channel, multiplier in holiday_multipliers.items():
        df.loc[df['is_holiday_period'], channel] = df.loc[df['is_holiday_period'], channel] * multiplier

    # Create performance metrics based on spend but with diminishing returns

    # Helper function for diminishing returns
    def diminishing_returns(x, scale=100000, shape=0.7):
        return (x / scale) ** shape

    # Generate performance metrics with realistic relationships to spend
    df['tv_grps'] = 100 * diminishing_returns(df['tv_spend'], 100000) + add_noise(num_periods, 5)
    df['digital_impressions'] = 1000000 * diminishing_returns(df['digital_display_spend'], 50000) + add_noise(
        num_periods, 50000)
    df['digital_clicks'] = df['digital_impressions'] * (0.02 + 0.01 * np.random.random(num_periods))
    df['search_impressions'] = 500000 * diminishing_returns(df['search_spend'], 40000) + add_noise(num_periods, 25000)
    df['search_clicks'] = df['search_impressions'] * (0.05 + 0.02 * np.random.random(num_periods))
    df['social_impressions'] = 2000000 * diminishing_returns(df['social_media_spend'], 30000) + add_noise(num_periods,
                                                                                                          100000)
    df['social_engagements'] = df['social_impressions'] * (0.03 + 0.015 * np.random.random(num_periods))
    df['email_opens'] = 50000 * diminishing_returns(df['email_spend'], 10000) + add_noise(num_periods, 5000)
    df['email_clicks'] = df['email_opens'] * (0.1 + 0.05 * np.random.random(num_periods))

    # Add control variables
    df['price_index'] = 100 + add_trend(num_periods, 0.05) + 5 * add_seasonality(num_periods, cycles=2) + add_noise(
        num_periods, 2)
    df['competitor_price_index'] = 102 + add_trend(num_periods, 0.08) + 3 * add_seasonality(num_periods,
                                                                                            cycles=1.7) + add_noise(
        num_periods, 3)
    df['promotion_flag'] = np.random.random(num_periods) < 0.3
    df['gdp_index'] = 100 + add_trend(num_periods, 0.1) + 2 * add_seasonality(num_periods, cycles=1) + add_noise(
        num_periods, 1)
    df['consumer_confidence'] = 70 + add_trend(num_periods, 0.05) + 10 * add_seasonality(num_periods,
                                                                                         cycles=1.2) + add_noise(
        num_periods, 5)
    df['market_size'] = 10000000 + 2000000 * add_seasonality(num_periods, cycles=1) + add_noise(num_periods, 500000)

    # Create outcome variables with known relationships to marketing variables

    # Define true elasticities for each channel
    elasticities = {
        'tv_spend': 0.2,
        'digital_display_spend': 0.15,
        'search_spend': 0.25,
        'social_media_spend': 0.18,
        'video_spend': 0.12,
        'email_spend': 0.05
    }

    # Create adstock transformations (lagged effects)
    adstocked_spend = {}

    for channel, elasticity in elasticities.items():
        # Set channel-specific decay parameters
        if channel == 'tv_spend':
            decay_rate, max_lag = 0.85, 8  # TV has longer effect
        elif channel == 'search_spend':
            decay_rate, max_lag = 0.3, 3  # Search has immediate effect
        elif channel == 'social_media_spend':
            decay_rate, max_lag = 0.6, 5  # Medium decay
        else:
            decay_rate, max_lag = 0.7, 4  # Default

        # Apply adstock transformation
        spend_values = df[channel].values
        transformed = np.zeros_like(spend_values, dtype=float)
        transformed[0] = spend_values[0]

        for t in range(1, len(spend_values)):
            transformed[t] = spend_values[t]
            for lag in range(1, min(t + 1, max_lag + 1)):
                transformed[t] += spend_values[t - lag] * (decay_rate ** lag)

        # Store the transformed values
        adstocked_spend[channel] = transformed

    # Create base revenue with trend and seasonality
    base_revenue = 2000000 + add_trend(num_periods, 1000) + 500000 * add_seasonality(num_periods, cycles=1)

    # Add control variable effects
    price_effect = -2000 * (df['price_index'] - 100)
    competitor_effect = 1000 * (df['competitor_price_index'] - 100)
    promotion_effect = df['promotion_flag'].astype(int) * 100000
    economy_effect = 5000 * (df['gdp_index'] - 100) + 2000 * (df['consumer_confidence'] - 70)

    # Add marketing effects (with adstock and diminishing returns)
    marketing_effect = 0
    for channel, elasticity in elasticities.items():
        # Apply diminishing returns to adstocked values
        effect = 500000 * elasticity * diminishing_returns(adstocked_spend[channel], scale=np.mean(df[channel]) * 5)
        marketing_effect += effect

    # Combine all effects
    df[
        'revenue'] = base_revenue + price_effect + competitor_effect + promotion_effect + economy_effect + marketing_effect + add_noise(
        num_periods, 100000)

    # Generate secondary KPIs derived from revenue
    avg_price = 50 + add_noise(num_periods, 2)
    df['units_sold'] = df['revenue'] / avg_price
    df['new_customers'] = df['units_sold'] * (0.1 + 0.05 * np.random.random(num_periods))
    df['website_visits'] = df['units_sold'] * (10 + 2 * np.random.random(num_periods))
    df['conversion_rate'] = df['units_sold'] / df['website_visits']

    # Round numeric columns to appropriate precision
    for col in df.columns:
        if col.endswith('_spend') or col == 'revenue' or col == 'market_size':
            df[col] = df[col].round(2)
        elif col in ['tv_grps', 'price_index', 'competitor_price_index', 'gdp_index', 'consumer_confidence']:
            df[col] = df[col].round(2)
        elif col == 'conversion_rate':
            df[col] = df[col].round(4)
        elif col not in ['date', 'is_holiday_period', 'promotion_flag']:
            df[col] = df[col].astype(int)

    # Convert boolean columns to proper boolean type
    df['is_holiday_period'] = df['is_holiday_period'].astype(bool)
    df['promotion_flag'] = df['promotion_flag'].astype(bool)

    return df


def create_sql_dump(df, table_name='marketing_data', file_path='marketing_data.sql'):
    """
    Create SQL dump file for PostgreSQL from DataFrame

    Args:
        df: DataFrame with marketing data
        table_name: Name of the database table
        file_path: Path to save the SQL file
    """
    # Start with the CREATE TABLE statement
    sql_types = {
        'date': 'DATE',
        'year': 'INTEGER',
        'quarter': 'INTEGER',
        'month': 'INTEGER',
        'week_of_year': 'INTEGER',
        'is_holiday_period': 'BOOLEAN',
        'tv_spend': 'NUMERIC(12,2)',
        'digital_display_spend': 'NUMERIC(12,2)',
        'search_spend': 'NUMERIC(12,2)',
        'social_media_spend': 'NUMERIC(12,2)',
        'video_spend': 'NUMERIC(12,2)',
        'email_spend': 'NUMERIC(12,2)',
        'tv_grps': 'NUMERIC(8,2)',
        'digital_impressions': 'INTEGER',
        'digital_clicks': 'INTEGER',
        'search_impressions': 'INTEGER',
        'search_clicks': 'INTEGER',
        'social_impressions': 'INTEGER',
        'social_engagements': 'INTEGER',
        'email_opens': 'INTEGER',
        'email_clicks': 'INTEGER',
        'revenue': 'NUMERIC(14,2)',
        'units_sold': 'INTEGER',
        'new_customers': 'INTEGER',
        'website_visits': 'INTEGER',
        'conversion_rate': 'NUMERIC(5,4)',
        'price_index': 'NUMERIC(6,2)',
        'competitor_price_index': 'NUMERIC(6,2)',
        'promotion_flag': 'BOOLEAN',
        'gdp_index': 'NUMERIC(6,2)',
        'consumer_confidence': 'NUMERIC(6,2)',
        'market_size': 'NUMERIC(12,2)'
    }

    # Create table SQL
    create_table_sql = f"CREATE TABLE {table_name} (\n"
    for col in df.columns:
        create_table_sql += f"    {col} {sql_types[col]}"
        if col == 'date':
            create_table_sql += " PRIMARY KEY"
        create_table_sql += ",\n"
    create_table_sql = create_table_sql.rstrip(",\n") + "\n);\n\n"

    # Start SQL file
    with open(file_path, 'w') as f:
        f.write(create_table_sql)

        # Add indexes
        f.write(f"CREATE INDEX idx_{table_name}_date ON {table_name}(date);\n")
        f.write(f"CREATE INDEX idx_{table_name}_year_quarter ON {table_name}(year, quarter);\n\n")

        # Begin transaction
        f.write("BEGIN;\n\n")

        # Create INSERT statements
        batch_size = 100  # Insert in batches for better performance
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            values = []

            for _, row in batch.iterrows():
                row_values = []
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        row_values.append("NULL")
                    elif isinstance(value, (bool)):
                        row_values.append(str(value).upper())
                    elif isinstance(value, (int, float)):
                        row_values.append(str(value))
                    elif isinstance(value, pd.Timestamp):
                        row_values.append(f"'{value.strftime('%Y-%m-%d')}'")
                    else:
                        row_values.append(f"'{value}'")

                values.append("(" + ", ".join(row_values) + ")")

            insert_stmt = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES\n"
            insert_stmt += ",\n".join(values) + ";\n\n"
            f.write(insert_stmt)

        # Commit transaction
        f.write("COMMIT;\n")


def visualize_marketing_data(df, output_dir='plots'):
    """
    Create visualizations of the marketing data to validate patterns

    Args:
        df: DataFrame with marketing data
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot marketing spend over time
    plt.figure(figsize=(14, 7))
    for channel in ['tv_spend', 'digital_display_spend', 'search_spend',
                    'social_media_spend', 'video_spend', 'email_spend']:
        plt.plot(df['date'], df[channel], label=channel.replace('_', ' ').title())

    plt.title('Marketing Spend by Channel Over Time')
    plt.xlabel('Date')
    plt.ylabel('Spend ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'marketing_spend.png'))
    plt.close()

    # Plot revenue trend
    plt.figure(figsize=(14, 7))
    plt.plot(df['date'], df['revenue'], label='Revenue', color='blue')

    # Add recession bands if relevant
    plt.title('Revenue Over Time')
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'revenue_trend.png'))
    plt.close()

    # Plot correlation matrix of key variables
    key_vars = ['tv_spend', 'digital_display_spend', 'search_spend',
                'social_media_spend', 'video_spend', 'email_spend',
                'price_index', 'competitor_price_index', 'gdp_index',
                'consumer_confidence', 'revenue']

    corr = df[key_vars].corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.columns)), corr.columns)

    # Add correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                            ha='center', va='center', color='black')

    plt.title('Correlation Matrix of Key Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    # Plot seasonality patterns in revenue
    plt.figure(figsize=(14, 7))

    # Group by month and calculate average
    monthly_avg = df.groupby('month')['revenue'].mean()
    months = list(range(1, 13))

    plt.bar(months, monthly_avg.values)
    plt.title('Average Revenue by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Revenue ($)')
    plt.xticks(months)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'revenue_seasonality.png'))
    plt.close()

    # Plot price vs units sold
    plt.figure(figsize=(10, 7))
    plt.scatter(df['price_index'], df['units_sold'], alpha=0.6)
    plt.title('Price Impact on Units Sold')
    plt.xlabel('Price Index')
    plt.ylabel('Units Sold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_impact.png'))
    plt.close()


if __name__ == "__main__":
    # Generate the mock data
    print("Generating mock marketing data...")
    mock_data = generate_mock_mmm_data(start_date='2020-01-01', end_date='2023-12-31')

    # Save to CSV
    csv_path = '../data/mock_marketing_data.csv'
    mock_data.to_csv(csv_path, index=False)
    print(f"Mock data saved to {csv_path}")

    # Create SQL dump
    sql_path = '../data/marketing_data.sql'
    create_sql_dump(mock_data, file_path=sql_path)
    print(f"SQL dump saved to {sql_path}")

    # Visualize data
    print("Creating visualizations...")
    visualize_marketing_data(mock_data)
    print("Visualizations saved to 'plots' directory")

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Time Range: {mock_data['date'].min()} to {mock_data['date'].max()}")
    print(f"Number of Observations: {len(mock_data)}")
    print(
        f"Total Marketing Spend: ${mock_data[['tv_spend', 'digital_display_spend', 'search_spend', 'social_media_spend', 'video_spend', 'email_spend']].sum().sum():,.2f}")
    print(f"Total Revenue: ${mock_data['revenue'].sum():,.2f}")

    # Print data sample
    print("\nData Sample:")
    print(mock_data.head())