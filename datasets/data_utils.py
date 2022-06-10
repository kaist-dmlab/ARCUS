import pandas as pd
import numpy as np
import tensorflow as tf

def load_dataset(args):
    dataset = pd.read_csv("datasets/"+args.dataset_name+".csv", dtype=np.float64, header=None)
    dataset_label = dataset.pop(dataset.columns[-1])
    args.input_dim = tf.data.Dataset.from_tensor_slices((dataset.values, dataset_label.values)).element_spec[0].shape[0]
    return args, tf.data.Dataset.from_tensor_slices((dataset.values, dataset_label.values))
