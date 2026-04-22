import pandas as pd

df = pd.read_csv('../u.data', sep='\t', names=['user_id','item_id','rating','timestamp'])

df.to_csv('../movielens.csv', index=False)

print("CSV erstellt!")
