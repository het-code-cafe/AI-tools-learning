# Remove rows with any missing values
df_cleaned = df.dropna()

# Remove rows with missing values in a specific column
df_cleaned = df.dropna(subset=['column_name'])

# Fill missing values with a specific value (e.g., 0)
df_filled = df.fillna(0)

# Fill missing values with the mean of the column
df_filled = df.fillna(df.mean())