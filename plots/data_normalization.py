import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sample Data
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
print("Min-Max Scaling:\n", df_minmax)

# Z-Score Normalization
z_score_scaler = StandardScaler()
df_zscore = pd.DataFrame(z_score_scaler.fit_transform(df), columns=df.columns)
print("\nZ-Score Normalization:\n", df_zscore)

# Decimal Scaling Normalization
# Calculate the scaling factor for decimal scaling
scaling_factors = 10 ** df.abs().max().astype(str).apply(lambda x: len(x.split('.')[0]))

df_decimal_scaled = df / scaling_factors
print("\nDecimal Scaling Normalization:\n", df_decimal_scaled)
