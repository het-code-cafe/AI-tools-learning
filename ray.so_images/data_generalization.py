import pandas as pd

# Sample DataFrame with a date column
data = {'date': ['2023-01-15', '2023-02-20', '2023-03-25', '2023-04-30']}
df = pd.DataFrame(data)

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Extract the month from the date column
df['month'] = df['date'].dt.month

# If you want the month name instead of the number, use:
df['month_name'] = df['date'].dt.month_name()

print(df)
