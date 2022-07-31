import abc
import tensorflow as tf
from tensorflow.keras import Model

class BaseModel(Model, metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def get_latent(self, x, training = False):
        pass

    @abc.abstractmethod
    def train_step(self, x):
        pass

    @abc.abstractmethod
    def inference_step(self, x):
        pass 