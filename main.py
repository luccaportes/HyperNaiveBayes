import pandas as pd
from nb import NB
from sklearn.preprocessing import OneHotEncoder

nb = NB()

# df = pd.read_csv("db.txt")
#
# x = df.iloc[:, :-1]
# y = df.iloc[:, -1]
#

#
# nb.fit(x, y)
#
# nb.predict(x[:1])

# print()

enc = OneHotEncoder(handle_unknown='ignore')
df = pd.read_csv("tennis.txt")
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(x)
print(y)
x = enc.fit_transform(x, y)

print(x)
print(y)

