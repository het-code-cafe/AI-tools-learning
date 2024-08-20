import pandas as pd

# Sample IQ data
df = pd.DataFrame({'IQ': [85, 90, 95, 100, 105, 110, 115, 120, 125, 130]})

# Binning IQ scores into categories
df['IQ_Level'] = pd.cut(df['IQ'], bins=[0, 90, 110, 200], labels=["Below Average", "Average", "Above Average"], right=False)

print(df)
