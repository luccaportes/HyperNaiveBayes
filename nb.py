import numpy as np
import pandas as pd
import collections as collec


class NB:
    def __init__(self):
        self.dict_per_class = {}
        self.stats_per_class = collec.defaultdict(dict)

    def fit(self, x, y):
        classes = np.unique(y)
        for c in classes:
            indexes = y == c
            c_x = x[indexes]
            self.dict_per_class[c] = c_x

        for label, feats in self.dict_per_class.items():
            n_inst = len(feats)
            for col in feats.columns:
                self.stats_per_class[label][col] = (feats[col].sum(), feats[col].sum()/n_inst)
        print(self.stats_per_class)

    def predict(self, x):
        existing_cols = []
        for col in x.columns:
            if x[col].tolist()[0] == 1:
                existing_cols.append(col)
        prob_cls = {}
        for cls, stats in self.stats_per_class.items():
            prob = 1
            for feat in existing_cols:
                prob *= stats[feat][1]
            n_instances = prob * len(self.dict_per_class[cls])
            prob_cls[cls] = n_instances / len(self.dict_per_class[cls])
        print(prob_cls)


