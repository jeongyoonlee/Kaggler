from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import unittest
from kaggler.online_model import SGD


DUMMY_SPARSE_STR = """0 1:1 3:1 10:1
0 3:1 5:1
1 4:1 6:1 8:1 10:1"""

DUMMY_Y = [0, 0, 1]
DUMMY_LEN_X = [3, 2, 4]


class TestSGD(unittest.TestCase):

    def setUp(self):
        self.model = SGD(n=2**10, a=0.1, l1=1, l2=1, interaction=True)
        self.sparse_file = '/tmp/dummy.sps'

        """Create dummpy sparse files."""
        with open(self.sparse_file, 'w') as f:
            f.write(DUMMY_SPARSE_STR)

    def tearDown(self):
        # If a dummy file exists, remove it.
        if os.path.isfile(self.sparse_file):
            os.remove(self.sparse_file)

    def test_read_sparse(self):
        len_xs = []
        ys = []
        for x, y in self.model.read_sparse(self.sparse_file):
            # check hash collision for feature index
            self.assertEqual(len(set(x)), len(x))

            ys.append(y)
            len_xs.append(len(x))

        # check if target values are correct
        self.assertEqual(ys, DUMMY_Y)

        # check if the number of feature index are correct
        self.assertEqual(len_xs, DUMMY_LEN_X)


if __name__ == '__main__':
    unittest.main()

