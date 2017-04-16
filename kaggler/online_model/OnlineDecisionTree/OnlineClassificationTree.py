from _tree import Tree
from OnlineDecisionTree import *
from utils import *
import numpy as np
import pandas as pd

class ClassificationTree(Tree):

    def __init__(
            self,
            number_of_features,
            number_of_functions=10,
            min_sample_split=200,
            predict_initialize={
                'count_dict': {},
            }
        ):
        # Constant values
        self.number_of_features = number_of_features
        self.number_of_functions = number_of_functions
        self.min_sample_split = min_sample_split
        self.predict_initialize = predict_initialize
        self.max_sample = 1000
        # Dynamic values
        self.left = None
        self.right = None
        self.randomly_selected_features = []
        self._randomly_select()
        self.criterion = None


    def _calculate_split_score(self, split):
        """
        calculate the score of the split:
        score = current_error - after_split_error
        """
        left_error = gini(split['left'])
        right_error = gini(split['right'])
        error = gini(self.Y)
        # if the split is any good, the score should be greater than 0
        total = float(len(self.Y))
        score = error - 1 / total * (len(split['left']) * left_error\
                                     + len(split['right']) * right_error)
        return score

    def _apply_best_split(self):
        best_split, best_split_score = self._find_best_split()
        if best_split_score > 0:
            self.criterion = lambda x : x[best_split['feature']] \
                             > best_split['value']
            # create the left child
            self.left = ClassificationTree(
                number_of_features=self.number_of_features,
                number_of_functions=self.number_of_functions,
                min_sample_split=self.min_sample_split,
                predict_initialize={
                    'count_dict': count_dict(best_split['left']),
                }
            )
            # create the right child
            self.right = ClassificationTree(
                number_of_features=self.number_of_features,
                number_of_functions=self.number_of_functions,
                min_sample_split=self.min_sample_split,
                predict_initialize={
                    'count_dict': count_dict(best_split['right']),
                }
            )
            # Collect garbage
            self.samples = {}
            self.Y = []


    def predict(self, x):
        """
        Make prediction recursively. Use both the samples inside the current
        node and the statistics inherited from parent.
        """
        if self._is_leaf():
            d1 = self.predict_initialize['count_dict']
            d2 = count_dict(self.Y)
            for key, value in d1.iteritems():
                if key in d2:
                    d2[key] += value
                else:
                    d2[key] = value
            return argmax(d2)
        else:
            if self.criterion(x):
                return self.right.predict(x)
            else:
                return self.left.predict(x)
