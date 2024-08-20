import pandas as pd

# Sample sales data
data = {
    'date': ['2023-08-01', '2023-08-01', '2023-08-02', '2023-08-02', '2023-08-03'],
    'product': ['A', 'B', 'A', 'B', 'A'],
    'sales': [100, 150, 200, 250, 300]
}

df = pd.DataFrame(data)

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Aggregate the sales data by product
# Here, we'll calculate the total sales per product
aggregated_data = df.groupby('product').agg(total_sales=('sales', 'sum'))

print(aggregated_data)
