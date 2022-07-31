# ORIGINAL SOURCE: https://openreview.net/forum?id=HkgeGeBYDB
# @inproceedings{
# Kim2020RaPP:,
# title={RaPP: Novelty Detection with Reconstruction along Projection Pathway},
# author={Ki Hyun Kim and Sangwoo Shim and Yongsub Lim and Jongseob Jeon and Jeongwoo Choi and Byungchan Kim and Andre S. Yoon},
# booktitle={International Conference on Learning Representations},
# year={2020},
# url={https://openreview.net/forum?id=HkgeGeBYDB}
# }

import tensorflow as tf

from model.model_base import BaseModel
from tensorflow.keras.layers import Dense, Layer, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras import Model, Sequential, activations

def SAP(H):
    hidden_difference = []
    for hidden_activation_origin_x, hidden_activation_recons_x in H:
        l2_norm = tf.math.square(tf.norm(hidden_activation_origin_x - hidden_activation_recons_x, axis = 1))
        hidden_difference.append(l2_norm)
    
    SAP = 0
    for difference in hidden_difference:
        SAP += difference
    
    return SAP

def NAP(H):
    hidden_difference = []
    for hidden_activation_origin_x, hidden_activation_recons_x in H:
        hidden_difference.append(hidden_activation_origin_x - hidden_activation_recons_x)

    # matrix D
    D = tf.concat(hidden_difference, axis = 1) # N X (d1 + d2 + ...)
    U = tf.math.reduce_mean(D, axis = 0) # (d1 + d2 + ...)
    
    D_bar   = D - U  # N X (d1 + d2 + ...)
    s, u, v = tf.linalg.svd(D_bar, full_matrices = False)
    s = tf.where(tf.equal(s, 0), tf.ones_like(s), s)
    NAP = tf.norm(tf.linalg.matmul(tf.linalg.matmul(D_bar, v), tf.linalg.inv(tf.linalg.diag(s))), axis = 1)
    
    return NAP

class RAPP(BaseModel):
    def __init__(self,
                 hidden_layer_sizes,
                 learning_rate = 1e-4,
                 activation = 'relu',
                 bn = True,
                 name = 'RAPP'):
        super(RAPP, self).__init__(name = name)
        
        if activation == 'relu':
            activation = tf.nn.relu
        elif activation == 'leakyrelu':
            activation = tf.nn.leaky_relu
        else:
            print(f'This {activation} function is not allowed')
        
        self.encoder = []
        for idx, layer in enumerate(hidden_layer_sizes[1:]):
            self.encoder.append(Dense(layer, activation = 'linear', name = f"layer_{idx}"))
            self.encoder.append(Activation(activation))
            if bn == True:
                self.encoder.append(BatchNormalization(name = f"bn_encoder_{idx}"))
        
        self.decoder = []
        for idx, layer in enumerate(list(reversed(hidden_layer_sizes))[1:-1]):
            self.decoder.append(Dense(layer, activation = 'linear', name = f"layer_{idx}"))
            self.decoder.append(Activation(activation))
            if bn == True:
                self.decoder.append(BatchNormalization(name = f"bn_decoder_{idx}"))
        
        idx += 1
        self.decoder.append(Dense(list(reversed(hidden_layer_sizes))[-1], activation = 'linear', name = f"layer_{idx}"))
        
        self.encoder_layer = Sequential(self.encoder, name = 'encoder')
        self.decoder_layer = Sequential(self.decoder, name = 'decoder')
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.loss      = tf.keras.losses.MeanSquaredError()
        self.bn        = bn
        
        
    def call(self, x, training = False):
        latent_x = self.encoder_layer(x, training = training)
        recons_x = self.decoder_layer(latent_x, training = training)
        return recons_x
    
    def get_latent(self, x, training = False):
        latent_x = self.encoder_layer(x, training = training)
        return latent_x
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            recons_x = self.call(x, training = True)
            loss = self.loss(x, recons_x)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        
        return loss
    
    def get_hidden_set(self, x):
        
        origin_x = x
        recons_x = self.call(x)
        
        dense_layer = []
        activ_layer = []
        
        if self.bn == False:
            for idx, layer in enumerate(self.encoder):
                if idx % 2 == 0:
                    dense_layer.append(layer)
                else:
                    activ_layer.append(layer)
        else:
            for idx, layer in enumerate(self.encoder):
                if idx % 3 == 0:
                    dense_layer.append(layer)
                elif idx % 3 == 1:
                    activ_layer.append(layer)
                else:
                    continue
                    
        H = []
        temp_origin = origin_x
        temp_recons = recons_x
        
        for dense, activation in zip(dense_layer, activ_layer):
            
            hidden_activation_origin_x = activation(dense(temp_origin))
            hidden_activation_recons_x = activation(dense(temp_recons))
            
            H.append((hidden_activation_origin_x, hidden_activation_recons_x))
            
            temp_origin = hidden_activation_origin_x
            temp_recons = hidden_activation_recons_x
        
        self.H = H
        return H
        
    def inference_step(self, x):
        self.get_hidden_set(x)
        
        #SAP_value = SAP(self.H)
        NAP_value = NAP(self.H)
        #try:
        #    NAP_value = NAP(self.H)
        #except:
        #    NAP_value = SAP(self.H)
        
        return NAP_value.numpy()

