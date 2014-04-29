#!/bin/python
#pypng should be installed
#it will take a long time to convert all 300,000 test examples
import png
import os
import numpy as np
HOME=os.path.expanduser("~")
prefix=HOME+"/DATA/kaggle/cifar10/"

def _flat2boxed(row):
    # Note we skip every 4th element, thus eliminating the alpha channel
    return [tuple(row[i:i+3]) for i in range(0, len(row), 3)]

def png2array(path):
    (w, h, p, m) = png.Reader(filename = path).asRGB8()
    img = np.asarray([_flat2boxed(r) for r in p], dtype=np.uint8)
    data = img[...,0].reshape(1024)
    data = np.concatenate((data, img[...,1].reshape(1024)), axis = 0)
    data = np.concatenate((data, img[...,2].reshape(1024)), axis = 0)
    return data.astype(np.uint8)

def work():
    num=300000
    #num=300
    step=10000
    for k in xrange(30):
        data=np.asarray([png2array(path="{}test/{}.png".format(prefix,i+1)) for i in xrange(step*k, step*(k+1))])
        print data.shape
        np.save(prefix+'data_{}.npy'.format(k), data)

if __name__ == '__main__':
    work()


