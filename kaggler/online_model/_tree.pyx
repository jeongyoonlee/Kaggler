import random
import numpy as np
cimport numpy as np


ctypedef np.int_t DTYPE_t


cpdef tuple bin_split(list sample_feature, double feature_value):
    cdef list left, right
    cdef tuple x
    left = [x[1] for x in sample_feature if x[0]<=feature_value]
    right = [x[1] for x in sample_feature if x[0]>feature_value]
    return left, right


cdef class Tree:

    def __cinit__(
            self,
            int number_of_features,
            int number_of_functions=10,
            int min_sample_split=20,
            dict predict_initialize={
                'mean':2.0,
                'variance':1.0,
                'num_samples':0
            }
        ):
        # Constant values
        self.number_of_features = number_of_features
        self.number_of_functions = number_of_functions
        self.min_sample_split = min_sample_split
        self.predict_initialize = predict_initialize
        self.max_sample = 100
        # Dynamic values
        self.left = None
        self.right = None
        self.randomly_selected_features = []
        self._randomly_select()
        self.criterion = None


    def _randomly_select(self):
        # Check the number of randomly selected features
        if self.number_of_features < self.number_of_functions:
            raise Exception("The feature number is more than maximum")

        # Randomly select features into a set, and then transform to a list
        self.randomly_selected_features=set([])
        while len(self.randomly_selected_features) < self.number_of_functions:
            self.randomly_selected_features.add(\
                random.randint(0, self.number_of_features-1))
        self.randomly_selected_features = list(self.randomly_selected_features)

        # Initialize the samples belong to the node
        self.samples = {}
        self.Y = []
        for feature in self.randomly_selected_features:
            self.samples[feature] = []

    def _is_leaf(self):
        return self.criterion == None

    cpdef update(self, np.ndarray x, y):
        """
        Update the model according to a single (x, y) input.

        If the current node is a leaf, then update the samples of the
        current node.

        Else update its left or right node recursively according to the
        value of x.
        When the left and right child are created, they inherit mean and
        sample count information from the parent.
        """
        cdef int N
        if self._is_leaf():
            N = len(self.Y)
            if N <= self.max_sample:
                self._update_samples(x, y)
            if N == self.min_sample_split or N == 2 * self.min_sample_split:
                self._apply_best_split()

        else:
            if self.criterion(x):
                self.right.update(x, y)
            else:
                self.left.update(x, y)

    cpdef _update_samples(self, np.ndarray x, DTYPE_t y):
        cdef int feature
        for feature in self.randomly_selected_features:
            self.samples[feature].append((x[feature], y))
        self.Y.append(y)

    cpdef tuple _find_best_split(self):
        cdef dict best_split = {}
        cdef double best_split_score = 0
        cdef int feature
        cdef double value
        cdef DTYPE_t prediction
        cdef list sample_feature
        cdef list left, right
        cdef dict split
        cdef double split_score
        # Try all the selected features and values combination, find the best
        for feature in self.randomly_selected_features:
            for (value, prediction) in self.samples[feature]:
                sample_feature = self.samples[feature]
                left, right = bin_split(sample_feature, value)

                split = {
                    'left': left,
                    'right': right,
                    'value': value,
                    'feature': feature,
                }

                split_score = self._calculate_split_score(split)
                if split_score > best_split_score:
                    best_split = split
                    best_split_score = split_score

        return best_split, best_split_score
