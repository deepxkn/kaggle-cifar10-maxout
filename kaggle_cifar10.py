"""
.. todo::

    WRITEME
"""
import os, cPickle, logging
_logger = logging.getLogger(__name__)

import numpy as np
N = np
from pylearn2.datasets import dense_design_matrix
from pylearn2.expr.preprocessing import global_contrast_normalize

HOME=os.path.expanduser("~")
prefix=HOME+"/DATA/kaggle/cifar10/"
file=prefix+"data300000.npy"

class CIFAR10_TEST(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, center = False, rescale = False, gcn = None,
            one_hot = False, start = None, stop = None, axes=('b', 0, 1, 'c'),
            toronto_prepro = False, preprocessor = None):
        # note: there is no such thing as the cifar10 validation set;
        # pylearn1 defined one but really it should be user-configurable
        # (as it is here)

        self.axes = axes

        # we define here:
        dtype  = 'uint8'
        ntrain = 0
        nvalid = 0  # artefact, we won't use it
        ntest  = 300000

        # we also expose the following details:
        self.img_shape = (3,32,32)
        self.img_size = N.prod(self.img_shape)
        #self.n_classes = 10
        #self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog','horse','ship','truck']


        # prepare loading
        #fnames = ['data_batch_%i' % i for i in range(1,6)]
        #lenx = N.ceil((ntrain + nvalid) / 10000.)*10000
        #x = N.zeros((lenx,self.img_size), dtype=dtype)
        #y = N.zeros(lenx, dtype=dtype)
        X=np.load(file).astype(np.float32)
        # load train data

        if center:
            X -= 127.5
        self.center = center

        if rescale:
            X /= 127.5
        self.rescale = rescale

        self.toronto_prepro = toronto_prepro

        self.gcn = gcn
        if gcn is not None:
            gcn = float(gcn)
            X = global_contrast_normalize(X, scale=gcn)

        view_converter = dense_design_matrix.DefaultViewConverter((32,32,3), axes)

        super(CIFAR10, self).__init__(X=X, view_converter=view_converter)

        assert not np.any(np.isnan(self.X))

        if preprocessor:
            preprocessor.apply(self)

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        #assumes no preprocessing. need to make preprocessors mark the new ranges
        rval = X.copy()

        #patch old pkl files
        if not hasattr(self,'center'):
            self.center = False
        if not hasattr(self,'rescale'):
            self.rescale = False
        if not hasattr(self,'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            for i in xrange(rval.shape[0]):
                rval[i,:] /= np.abs(rval[i,:]).max()
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval,-1.,1.)

        return rval

    def adjust_to_be_viewed_with(self, X, orig, per_example = False):
        """
        .. todo::

            WRITEME
        """
        # if the scale is set based on the data, display X oring the scale determined
        # by orig
        # assumes no preprocessing. need to make preprocessors mark the new ranges
        rval = X.copy()

        #patch old pkl files
        if not hasattr(self,'center'):
            self.center = False
        if not hasattr(self,'rescale'):
            self.rescale = False
        if not hasattr(self,'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            if per_example:
                for i in xrange(rval.shape[0]):
                    rval[i,:] /= np.abs(orig[i,:]).max()
            else:
                rval /= np.abs(orig).max()
            rval = np.clip(rval, -1., 1.)
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval,-1.,1.)

        return rval

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return CIFAR10(which_set='test', center=self.center, rescale=self.rescale, gcn=self.gcn,
                one_hot=self.one_hot, toronto_prepro=self.toronto_prepro, axes=self.axes)


    @classmethod
    def _unpickle(cls, file):
        """
        .. todo::

            What is this? why not just use serial.load like the CIFAR-100
            class? Whoever wrote it shows up as "unknown" in git blame.
        """
        from pylearn2.utils import string_utils
        fname = os.path.join(
                string_utils.preprocess('${PYLEARN2_DATA_PATH}'),
                'cifar10',
                'cifar-10-batches-py',
                file)
        if not os.path.exists(fname):
            raise IOError(fname+" was not found. You probably need to download "
                    "the CIFAR-10 dataset by using the download script in pylearn2/scripts/datasets/download_cifar10.sh "
                    "or manually from http://www.cs.utoronto.ca/~kriz/cifar.html")
        _logger.info('loading file %s' % fname)
        fo = open(fname, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict
