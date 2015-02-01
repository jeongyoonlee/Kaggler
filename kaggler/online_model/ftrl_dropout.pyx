from csv import DictReader
from math import exp, log, sqrt

import cPickle as pickle
import gzip
import random


class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction=False, dropout=1.0):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction
        self.dropout = dropout        

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        
        self.w = [0.] * D  # use this for execution speed up

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        for i in x:
            yield i

        if self.interaction:
            L = len(x)
            for i in xrange(1, L):  # skip bias term, so we start at 1
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % self.D

    def predict(self, x, dropped = None):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''
        # params
        dropout = self.dropout

        # model
        w = self.w

        # wTx is the inner product of w and x
        wTx = 0.
        for j, i in enumerate(self._indices(x)):
            
            if dropped != None and dropped[j]:
                continue
           
            wTx += w[i]
        
        if dropped != None: wTx /= dropout 

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = self.w  # no need to change this, it won't gain anything
        dropout = self.dropout

        ind = [ i for i in self._indices(x)]
        
        if dropout == 1:
            dropped = None
        else:
            dropped = [random.random() > dropout for i in xrange(0,len(ind))]
        
        p = self.predict(x, dropped)

        # gradient under logloss
        g = p - y

        # update z and n
        for j, i in enumerate(ind):

            # implement dropout as overfitting prevention
            if dropped != None and dropped[j]: continue

            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g
            
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights -
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

    def read_csv(self, f_train):
        ''' GENERATOR: Apply hash-trick to the original csv row
                       and for simplicity, we one-hot-encode everything

            INPUT:
                path: path to training or testing file

            YIELDS:
                ID: id of the instance, mainly useless
                x: a list of hashed and one-hot-encoded 'indices'
                   we only need the index since all values are either 0 or 1
                y: y = 1 if we have a click, else we have y = 0
        '''
        for t, row in enumerate(DictReader(f_train)):
            # process id
            ID = row['id']
            del row['id']

            # process clicks
            y = 0.
            if 'click' in row:
                if row['click'] == '1':
                    y = 1.
                del row['click']
     
            # turn hour really into hour, it was originally YYMMDDHH

            date = row['hour'][0:6]
            row['hour'] = row['hour'][6:]
            
            #       stderr.write("_%s_" % date)
            
            # extract date
            row['wd'] = str(int(date) % 7)
            row['wd_hour'] = "%s_%s" % (row['wd'], row['hour'])            

            # build x
            x = [0]  # 0 is the index of the bias term
            for key in row:
                value = row[key]

                # one-hot encode everything with hash trick
                index = abs(hash(key + '_' + value)) % self.D
                x.append(index)

            yield t, ID, x, y

    def write_model(self, model, model_save, args):
       with gzip.open(model_save, "wb") as model_file:
           pickle.dump((args, model), model_file)

    def load_model(self, model_save):
        with gzip.open(model_save, "rb") as model_file:
            (p, model) = pickle.load(model_file)
        
        return model
