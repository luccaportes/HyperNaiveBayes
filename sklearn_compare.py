from sklearn.naive_bayes import GaussianNB
import pandas as pd

df_zao = pd.read_csv("tennis.txt").sample(frac=1, random_state=123)
df_train = df_zao[:10]
df_test = df_zao[10:]

x_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
x_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]


nb = GaussianNB()

nb.fit(x_train, y_train)

print(nb.score(x_test, y_test))