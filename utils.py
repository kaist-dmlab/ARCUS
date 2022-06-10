import os
import tensorflow as tf
import random
import numpy as np

def set_gpu(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if tf.test.is_built_with_cuda():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    return args 

def set_seed(seed):
    tf.keras.backend.set_floatx('float64')
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)