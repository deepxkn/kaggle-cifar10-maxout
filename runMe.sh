
#export THEANO_FLAGS='base_compiledir=/tmp/theano/kaggle-cifar10/,device=gpu1,floatX=float32'

THEANO_FLAGS="device=gpu1,floatX=float32,exception_verbosity=high" python make_predict.py ~/pylearn2/pylearn2/scripts/papers/maxout/predict_cifar10/cifar10_epoch149.pkl > ret.info
