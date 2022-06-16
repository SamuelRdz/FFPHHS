from PBC4cip import *
import pandas as pd
from PBC4cip.core.Helpers import get_col_dist, get_idx_val

df = pd.read_csv('train_set.csv')
features = ["EDGE_DENSITY", "AVG_DEGREE",
            "BURNING_NODES", "BURNING_EDGES", "NODES_IN_DANGER"]
df = df.sample(50)
print(df)

X_train = df[features]
y_train = df[["HEURISTIC"]]
# X_test = df_test[features]
# y_test = df_test[cla]

clf = PBC4cip(filtering=False)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print(y_pred)

predicted = y_pred
y = y_train
y_class_dist = get_col_dist(y[f'{y.columns[0]}'])
real = list(map(lambda instance: get_idx_val(
    y_class_dist, instance), y[f'{y.columns[0]}']))
print('y_class_dist', y_class_dist)
print('real', real)
