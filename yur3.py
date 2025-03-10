import pandas as pd
model_df = winners = pd.read_csv('all_correct.csv')
pd.set_option('display.max_rows', None)
model_df = model_df[model_df['Year'] == 2016]
model_df = model_df[model_df['Team'] == "Hawai’i"]
print(model_df)
