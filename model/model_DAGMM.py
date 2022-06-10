# ORIGINAl SOURCE (unofficial): https://github.com/tnakae/DAGMM

# @inproceedings{DAGMM,
#   title={Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection},
#   author={Zong, Bo and Song, Qi and Min, Martin Renqiang and Cheng, Wei and Lumezanu, Cristian and Cho, Daeki and Chen, Haifeng},
#   booktitle={International Conference on Learning Representations (ICLR)},
#   year={2018}
# }

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Sequential, optimizers
from tensorflow.keras.layers import Dense, Layer, Dropout, BatchNormalization

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc,roc_curve, precision_recall_curve, average_precision_score, precision_recall_fscore_support

class CompressionNet(Layer):
    def __init__(self,
                 hidden_layer_sizes,
                 activation,
                 bn = True,
                 name = 'CompressionNet'):
        super(CompressionNet, self).__init__(self, name = name)

        self.encoder = []
        for idx, size in enumerate(hidden_layer_sizes[1:]):
            self.encoder.append(Dense(size, activation = activation, name = f"layer_{idx}"))
            if bn == True:
                self.encoder.append(BatchNormalization(name = f"bn_{idx}"))
        
        self.decoder = []
        for idx, size in enumerate(list(reversed(hidden_layer_sizes))[1:-1]):
            self.decoder.append(Dense(size, activation = activation, name = f"layer_{idx}"))
            if bn == True:
                self.decoder.append(BatchNormalization(name = f"bn_{idx}"))
        
        idx += 1
        self.decoder.append(Dense(list(reversed(hidden_layer_sizes))[-1], activation = 'linear', name = f"layer_{idx}"))
        
        self.encoder_layer = Sequential(self.encoder, name = 'encoder')
        self.decoder_layer = Sequential(self.decoder, name = 'decoder')
        self.bn            = bn 
        
    def call(self, x, training = False):
        x_latent = self.encoder_layer(x, training = training)
        x_tilde = self.decoder_layer(x_latent, training = training)
        
        rec_cosine    = -tf.keras.losses.cosine_similarity(x, x_tilde, axis = 1)
        rec_euclidean = tf.divide(tf.norm(tensor = x - x_tilde, ord = 'euclidean', axis = 1), tf.norm(tensor = x, ord = 'euclidean', axis = 1))
        
        return x_latent, x_tilde, rec_cosine, rec_euclidean
    
    def reconstruction_error(self, x, x_tilde):
        return tf.math.reduce_mean(tf.math.reduce_sum(tf.math.square(x - x_tilde), axis = 1), axis = 0)

class EstimationNet(Layer):
    def __init__(self,
                 hidden_layer_sizes,
                 activation,
                 dropout_ratio = None,
                 name = "EstimationNet"):
        super(EstimationNet, self).__init__(name = name)
        
        self.estimation = []
        for idx, size in enumerate(hidden_layer_sizes[:-1]):
            self.estimation.append(Dense(size, activation = activation, name = f"layer_{idx}"))
            if dropout_ratio is not None:
                self.estimation.append(Dropout(rate = dropout_ratio, name = f"dropout_{idx}"))
                
        idx += 1
        self.estimation.append(Dense(hidden_layer_sizes[-1], activation = 'softmax', name = f"layer_softmax"))
        self.estimation_layer = Sequential(self.estimation, name = 'estimation')
        
    def call(self, z, training = False):
        gamma = self.estimation_layer(z, training = training)
        return gamma

class GMM(Layer):
    def __init__(self,
                 name = "GMM"):
        super(GMM, self).__init__(name = name)
        
        # gmm's parameter calculation
    def call(self, z, gamma):
        # Calculate phi, mu, sigma
        # i   : index of samples
        # k   : index of components
        # l,m : index of features
        gamma_sum = tf.reduce_sum(gamma, axis=0) 
        self.phi = tf.reduce_mean(gamma, axis=0) 
        self.mu  = tf.einsum('ik,il->kl', gamma, z) / tf.expand_dims(gamma_sum, axis = 1)
        z_centered = tf.math.sqrt(tf.expand_dims(gamma, axis = 2)) * (tf.expand_dims(z, axis = 1) - tf.expand_dims(self.mu, axis = 0))
        self.sigma = sigma = tf.einsum('ikl,ikm->klm', z_centered, z_centered)/tf.expand_dims(tf.expand_dims(gamma_sum, axis = 1), axis = 1)
        
        n_features = z.shape[1]
        min_vals = tf.linalg.diag(tf.ones(n_features, dtype = tf.float64)) * 1e-6
        self.L   = tf.linalg.cholesky(self.sigma + tf.expand_dims(min_vals, axis = 0))
        
    def energy(self, z):
        
        z_centered = tf.expand_dims(z, axis = 1) - tf.expand_dims(self.mu, axis = 0)
        v = tf.linalg.triangular_solve(self.L, tf.transpose(z_centered, [1, 2, 0]))  # kli
        
        log_det_sigma = 2.0 * tf.math.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L)), axis=1)
        
        d = z.get_shape().as_list()[1]
        logits = tf.math.log(tf.expand_dims(self.phi, axis = 1)) - 0.5 * tf.math.reduce_sum(tf.square(v), axis=1) + d * tf.math.log(2.0 * tf.cast(np.pi, dtype = tf.float64)) + tf.expand_dims(log_det_sigma, axis = 1)
        energies = -tf.math.reduce_logsumexp(logits, axis=0)
        
        return energies
    
    def cov_diag_loss(self):
        diag_loss = tf.math.reduce_sum(tf.divide(1, tf.linalg.diag_part(self.sigma)))
        return diag_loss
        
class DAGMM(Model):
    def __init__(self,
                 comp_hidden_layer_sizes,
                 comp_activation,
                 est_hidden_layer_sizes,
                 est_activation,
                 learning_rate = 1e-4,
                 bn = True,
                 est_dropout_ratio = 0.5,
                 lambda1 = 0.1,
                 lambda2 = 0.005,
                 name = 'DAGMM'):
        super(DAGMM, self).__init__(name = name)
        
        self.optimizer  = optimizers.Adam(learning_rate = learning_rate)
        
        self.compressionNet = CompressionNet(hidden_layer_sizes = comp_hidden_layer_sizes, activation = comp_activation, bn = True)
        self.encoder = self.compressionNet.encoder
        self.decoder = self.compressionNet.decoder
        
        self.estimationNet  = EstimationNet(hidden_layer_sizes = est_hidden_layer_sizes, activation = est_activation, dropout_ratio = est_dropout_ratio)
        self.gmm            = GMM()
                                  
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
    def call(self, x, training = False):
        x_latent, x_tilde, rec_cosine, rec_euclidean = self.compressionNet(x, training = training)
        rec_cosine = tf.reshape(rec_cosine, shape = [-1,1])
        rec_euclidean = tf.reshape(rec_euclidean, shape = [-1,1])
        z = tf.concat([x_latent, rec_cosine, rec_euclidean], axis = 1)
        gamma = self.estimationNet(z, training = training)
        
        return x_latent, x_tilde, z, gamma
   
    def get_latent(self, x, training = False):
        x_latent, _, _, _ = self.compressionNet(x, training = training)
        return x_latent

    def train_step(self, x):
        with tf.GradientTape() as tape:
            x_latent, x_tilde, z, gamma = self.call(x, training = True)
            self.gmm(z, gamma)
            energy = self.gmm.energy(z)
            cov_diag_loss = self.gmm.cov_diag_loss()
            loss = self.compressionNet.reconstruction_error(x, x_tilde) + self.lambda1 * tf.reduce_mean(energy) + self.lambda2 * cov_diag_loss
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        
        return loss
    
    def inference_step(self, x):
        
        x_latent, x_tilde, z, gamma = self.call(x)
        energy = self.gmm.energy(z)
        
        return energy
