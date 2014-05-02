
#export THEANO_FLAGS='base_compiledir=/tmp/theano/kaggle-cifar10/,device=gpu1,floatX=float32'

PATH_PARAM=.

THEANO_FLAGS="device=gpu0,floatX=float32" python make_predict.py PATH_PARAM/predict_cifar10/cifar10_epoch442.pkl 2>time_k20m.info
