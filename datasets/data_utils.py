import pandas as pd
import numpy as np
import tensorflow as tf

def load_dataset(args):
    dataset = pd.read_csv("datasets/"+args.dataset_name+".csv", dtype=np.float64, header=None)
    dataset_label = dataset.pop(dataset.columns[-1])
    loader = tf.data.Dataset.from_tensor_slices((dataset.values, dataset_label.values))
    args.input_dim = loader.element_spec[0].shape[0]
    return args, loader