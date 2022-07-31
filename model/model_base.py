import abc
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization

class BaseModel(Model, metaclass=abc.ABCMeta):
    """
    Base model for ARCUS
    If you want to implement your autoencoder(AE) models in the ARCUS framework,
    you can build your model by inheriting this BaseModel.

    Note that your AEs have to follow the below instructions
    1. You have to make your encoder and decoder as follows.
        1.1 self.encoder: a list for layers in encoder
        1.2 self.decoder: a list for layers in decoder

    2. You have to indicate layer names.
        2.1 tf.keras.layers.Dense: "layer..."
        2.2 tf.keras.layers.BatchNormalization: "bn..." 

    3. You have to implement three methods.
        3.1 get_latent 
            : To get latent vectors from AEs
            params
                x: input tensor
                training: train/eval mode
            returns
                latent_x: output tensor from encoder
        3.2 train_step 
            : To train AEs
            params
                x: input tensor 
            returns
                loss: loss value
        3.3 inference_step
            : To get anomaly scores from AEs
            params
                x: input tensor
            returns
                score: anomaly score
    """

    @abc.abstractmethod
    def get_latent(self, x, training = False):
        pass

    @abc.abstractmethod
    def train_step(self, x):
        pass

    @abc.abstractmethod
    def inference_step(self, x):
        pass 