# ORIGINAL SOURCE: https://github.com/dmzou/RSRAE
# @inproceedings{lai2020robust,
# title={Robust Subspace Recovery Layer for Unsupervised Anomaly Detection},
# author={Chieh-Hsin Lai and Dongmian Zou and Gilad Lerman},
# booktitle={International Conference on Learning Representations},
# year={2020},
# url={https://openreview.net/forum?id=rylb3eBtwr},
# }

import tensorflow as tf
import numpy as np

from model.model_base import BaseModel
from tensorflow.keras import optimizers, Model, activations, Sequential
from tensorflow.keras.layers import Layer, Dense, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Reshape
from sklearn.metrics import roc_auc_score, average_precision_score

class RSR(Layer):
    
    def __init__(self, intrinsic_size ,name = "rsr", **kwargs):
        
        super(RSR, self).__init__(name = name, **kwargs)
        self.flatten = Flatten()
        
        self.intrinsic_size = intrinsic_size
        
    def build(self, input_shape):
        self.A = self.add_weight("A", shape = [int(input_shape[-1]), self.intrinsic_size])
        
    def call(self, y):
        y = self.flatten(y)
        return tf.linalg.matmul(y, self.A), self.A
        
class Renormalization(Layer):
    
    def __init__(self, 
                 name = "renormalization",
                 **kwargs):
        super(Renormalization, self).__init__(name = name, **kwargs)
        
    def call(self, y):
        return tf.math.l2_normalize(y, axis = -1)

class RSRAE(BaseModel):
    
    def __init__(self, 
                 hidden_layer_sizes,
                 learning_rate = 1e-4,
                 bn = True,
                 intrinsic_size = 10,
                 activation = 'relu',
                 norm_type = "MSE", 
                 loss_norm_type = "MSE",
                 name = "RSRAE", **kwargs):
        
        super(RSRAE, self).__init__(name = name, **kwargs)

        # encoder, decoder 
        self.encoder = []
        for idx, layer in enumerate(hidden_layer_sizes[1:]):
            self.encoder.append(Dense(layer, activation = activation, name = f"layer_{idx}"))
            if bn == True:
                self.encoder.append(BatchNormalization(name = f"bn_layer_{idx}"))
            
        self.decoder = []
        for idx, layer in enumerate(list(reversed(hidden_layer_sizes))[1:-1]):
            self.decoder.append(Dense(layer, activation = activation, name = f"layer_{idx}"))
            if bn == True:
                self.decoder.append(BatchNormalization(name = f"bn_layer_{idx}"))
        idx += 1
        self.decoder.append(Dense(list(reversed(hidden_layer_sizes))[-1], activation = 'linear', name = f"layer_{idx}"))
        
        self.encoder_layer = Sequential(self.encoder, name = 'encoder')
        self.decoder_layer = Sequential(self.decoder, name = 'decoder')
        
        self.bn = bn
        self.loss_norm_type = loss_norm_type
        self.norm_type      = norm_type
        self.intrinsic_size = intrinsic_size
                
        self.optimizer1 = optimizers.Adam(learning_rate = learning_rate)
        self.optimizer2 = optimizers.Adam(learning_rate = 10 * learning_rate)
        self.optimizer3 = optimizers.Adam(learning_rate = 10 * learning_rate)
                
        self.rsr     = RSR(intrinsic_size = intrinsic_size)
        self.renormaliztion = Renormalization()

        
    def call(self, x, training = False):
        y = self.encoder_layer(x, training = training)
        y_rsr, self.A = self.rsr(y)
        
        z = self.renormaliztion(y_rsr)            
        x_tilde = self.decoder_layer(z, training = training)
        
        return y, y_rsr, z, x_tilde 
    
    def reconstruction_error(self, x, x_tilde):
        x = tf.reshape(x, shape = [x.shape[0], -1])
        x_tilde = tf.reshape(x_tilde, shape = [x_tilde.shape[0], -1])
        
        return tf.math.reduce_mean(tf.math.square(tf.norm(x-x_tilde, ord=2, axis=1)))
        
    
    def pca_error(self, y, z):
        z = tf.linalg.matmul(z, tf.transpose(self.A))
        
        return tf.math.reduce_mean(tf.math.square(tf.norm(y-z, ord=2, axis=1)))

    def proj_error(self):
        return tf.math.reduce_mean(tf.math.square(tf.linalg.matmul(tf.transpose(self.A), self.A) - tf.eye(self.intrinsic_size, dtype = tf.float64)))
    
    def reconstruction_loss(self, x, x_tilde):
        x = tf.reshape(x, shape = [x.shape[0], -1])
        x_tilde = tf.rehape(x_tilde, shape = [x_tilde.shape[0], -1])
        
        return tf.math.square(tf.norm(x-x_tilde, ord=2, axis=1))

    def get_latent(self, x):
        _, _, z, _  = self.call(x, training = False)
        return z
        #return tf.linalg.matmul(z, tf.transpose(self.A))
        
    def get_output(self, x):
        _, _, _, x_tilde  = self.call(x, training = False)
        return x_tilde
    
    # three loss returned
    def train_step(self, x):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
            y, y_rsr, z, x_tilde = self.call(x, training = True)
            y = tf.keras.layers.Flatten()(y)
            loss = self.reconstruction_error(x, x_tilde)
            proj_error = self.proj_error()
            pca_error = self.pca_error(y, y_rsr)
        
        gradients1 = tape1.gradient(loss ,self.trainable_weights)
        self.optimizer1.apply_gradients(zip(gradients1, self.trainable_weights))
        gradients2 = tape2.gradient(pca_error, self.A)
        self.optimizer2.apply_gradients(zip([gradients2], [self.A]))
        gradients3 = tape3.gradient(proj_error, self.A)
        self.optimizer3.apply_gradients(zip([gradients3], [self.A]))
        
        return loss
        
    # outlier score(= -cosine similarity) return 
    def inference_step(self, x):
        features = self.get_output(x)
        flat_output = np.reshape(features, (np.shape(x)[0], -1))
        flat_input = np.reshape(x, (np.shape(x)[0], -1))
        
        cosine_similarity = np.sum(flat_output * flat_input, -1) / (np.linalg.norm(flat_output, axis=-1) + 0.000001) / (np.linalg.norm(flat_input, axis=-1) + 0.000001)
        
        return -1 * cosine_similarity