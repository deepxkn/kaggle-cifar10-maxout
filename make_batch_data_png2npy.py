#!/bin/python
#pypng should be installed
import png
import os
import numpy as np
HOME=os.path.expanduser("~")
prefix=HOME+"/DATA/kaggle/cifar10/"

def _flat2boxed(row):
    # Note we skip every 4th element, thus eliminating the alpha channel
    return [tuple(row[i:i+3]) for i in range(0, len(row), 3)]

def png2array(path="/Users/eric/10000.png"):
    (w, h, p, m) = png.Reader(filename = path).asRGB8()
    img = np.asarray([_flat2boxed(r) for r in p], dtype=np.uint8)
    data = img[...,0].reshape(1024)
    data = np.concatenate((data, img[...,1].reshape(1024)), axis = 0)
    data = np.concatenate((data, img[...,2].reshape(1024)), axis = 0)
    return data.astype(np.uint8)
    #return [_flat2boxed(r) for r in p]
    #print z

def save_data(output='batch1.pkl', data=None):
    f = open(output, 'wb')
    cPickle.dump(data, f)
    f.close()
    print 'transform data successfully'
    return

def work():
    #num=300000
    num=300
    data=np.asarray([png2array(path="{}test/{}.png".format(prefix,i+1)) for i in range(num)])
    print data.shape
    np.save('data{}.npy'.format(num), data)

if __name__ == '__main__':
    work()


