from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_clipboard()
df.head()
df.plot(kind='scatter', x='Bill', y='tip',figsize=(12,5))
reg = LinearRegression()
reg.fit(df[['Bill']], df['tip'])
print('slope', reg.coef_)
print('intercept', reg.intercept_)
df['pred'] = reg.predict(df[['Bill']])
reg.score(df[['Bill']], df['tip']) * 100