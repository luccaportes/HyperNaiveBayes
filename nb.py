import numpy as np
import pandas as pd
import collections as collec


class NB:
    def __init__(self):
        self.dict_per_class = {}
        self.stats_per_class = collec.defaultdict(dict)

    def fit(self, x, y):
        # Split the instances per class
        classes = np.unique(y)
        for c in classes:
            indexes = y == c
            c_x = x[indexes]
            self.dict_per_class[c] = c_x

        # Compute the probability of each individual feature happen in each class
        # n_occurences / total_instances_class
        for label, feats in self.dict_per_class.items():
            n_inst = len(feats)
            for col in feats.columns:
                self.stats_per_class[label][col] = (feats[col].sum(), feats[col].sum()/n_inst)

    def predict(self, x):
        existing_cols = []
        # Check which features are different than zero
        for col in x.columns:
            if x[col].tolist()[0] == 1:
                existing_cols.append(col)
        prob_cls = {}
        # For each previously computed statistics for each class
        for cls, stats in self.stats_per_class.items():
            prob = 1
            # multiply the probability of coexisting features
            for feat in existing_cols:
                prob *= stats[feat][1]
            # Compute the probability of the set of features belonging to each class
            n_instances = prob * len(self.dict_per_class[cls])
            prob_cls[cls] = n_instances / len(self.dict_per_class[cls])
        # Select the biggest one and return
        max_prob = -1
        answer = None
        for k, v in prob_cls.items():
            if v > max_prob:
                answer = k
        return answer



