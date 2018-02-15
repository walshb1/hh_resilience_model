import pandas as pd

df = pd.read_csv('~/Desktop/mys0.csv')
df = pd.concat([df.head(2),df.tail(2)])

print(df.head())
