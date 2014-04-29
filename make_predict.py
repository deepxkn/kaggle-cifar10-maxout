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

from kaggle_cifar10 import CIFAR10_TEST

_, model_path = sys.argv

model = serial.load(model_path)

src = model.dataset_yaml_src
batch_size = 100
model.set_batch_size(batch_size)

assert src.find('train') != -1
train = yaml_parse.load(src)

import theano.tensor as T

Xb = model.get_input_space().make_batch_theano()
Xb.name = 'Xb'

ymf = model.fprop(Xb)
ymf.name = 'ymf'

from theano import function

yp = T.argmax(ymf,axis=1)
batch_y = function([Xb],[yp])

def make_predictions(X):
    yy = []
    assert X.shape[0] % batch_size == 0
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
            x_arg = train.get_topological_view(x_arg)
        print "x_arg shape: ", x_arg.shape
        yy.append(batch_y(x_arg))
    return yy

print "Loading preprocessor"
preprocessor=serial.load(HOME+"/DATA/image/cifar10/pylearn2_gcn_whitened/preprocessor.pkl")

#Num=30
Num=30
y_preds=[]
for k in xrange(Num):
    print "Loading the test data"
    test = CIFAR10_TEST(file="{}data_{}.npy".format(prefix, k), gcn = 55.)
    print "Preprocessing the test data"
    test.apply_preprocessor(preprocessor = preprocessor, can_fit = False)

    X = test.X.astype('float32')
    print X.shape
    print "Making predict"
    y_pred = make_predictions(X)
    print 'y_pred TYPE: ', type(y_pred)
    print len(y_pred), len(y_pred[0])
    y_preds += y_pred

print len(y_preds), len(y_preds[-1])
y_preds = np.asarray(y_preds).flatten()
print y_preds.shape

np.save('bak/y_preds.npy', y_preds)

import csv

label_names = [ 'airplane', 'automobile', 'bird',  'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
labels_dict = { i: x for i, x in enumerate( label_names ) }

writer = csv.writer( open( output_file, 'wb' ))
writer.writerow( [ 'id', 'label' ] )
counter = 1

for y in y_preds:
    label = labels_dict[y]
    writer.writerow( [ counter, label ] )
    counter += 1

assert( counter == 300001 )


