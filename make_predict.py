#-*- coding: utf-8 -*-
#!/bin/python
from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
import numpy as np

import os
import scipy.io
HOME=os.path.expanduser("~")
prefix=HOME+"/DATA/kaggle/cifar10/"

_, model_path = sys.argv

model = serial.load(model_path)

src = model.dataset_yaml_src
batch_size = 6000
model.set_batch_size(batch_size)

def load_data(file=path+"data300000.npy".format(SNR)):
    return np.load(file)

X=load_data()
X=X[ids]

X = X.astype('float32')

import theano.tensor as T

Xb = model.get_input_space().make_batch_theano()
Xb.name = 'Xb'

ymf = model.fprop(Xb)
ymf.name = 'ymf'

from theano import function

yp = T.argmax(ymf,axis=1)
batch_y = function([Xb],[yp])


def make_predict():
    yy = []
    #assert isinstance(X[0], (int, long))
    assert isinstance(batch_size, py_integer_types)
    print X.shape
    loop = X.shape[0]/batch_size
    if X.shape[0] % batch_size != 0 :
        loop += 1
    for i in xrange(loop):
        print i,
        x_arg = X[i*batch_size:(i+1)*batch_size,:]
        print len(x_arg)
        if Xb.ndim > 2:
            x_arg = test.get_topological_view(x_arg)
        yy.append(batch_y(x_arg) )
    return yy

y_pred = make_predict()

y_pred = np.hstack(y_pred).flatten()

np.save('y_pred.npy', y_pred)


