
#export THEANO_FLAGS='base_compiledir=/tmp/theano/kaggle-cifar10/,device=gpu1,floatX=float32'

THEANO_FLAGS="device=gpu0,floatX=float32" python make_predict.py ~/pylearn2/pylearn2/scripts/papers/maxout/predict_cifar10/cifar10_epoch411.pkl 2>time_GTX780.info
