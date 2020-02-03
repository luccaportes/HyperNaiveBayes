import pandas as pd
from nb import NB
from sklearn.preprocessing import OneHotEncoder
df_zao = pd.read_csv("tennis.txt").sample(frac=1, random_state=123)
df_train = df_zao[:10]
df_test = df_zao[10:]

x_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
x_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

nb = NB()

nb.fit(x_train, y_train)

hits = 0

for i in range(len(x_test)):
    pred = (nb.predict(x_test.iloc[[i]]))
    if pred == y_test.iloc[i]:
        hits += 1
print(hits/len(x_test))

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
