import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)

penguins = sns.load_dataset("penguins")
penguins.dropna(how="any")
encoded_penguins = pd.get_dummies(penguins, columns=['species', 'island', 'sex'])

print(encoded_penguins.sample(n=10))