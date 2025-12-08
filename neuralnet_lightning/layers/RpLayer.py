
__all__ = ['RpLayer','rvec']


from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from copy import copy



rvec = np.concatenate( ( 
                       np.arange(1,9), # PS
                       np.arange(1,65),# EM1
                       np.arange(1,9), # EM2
                       np.arange(1,9), # EM3
                       np.arange(1,5), # HAD1
                       np.arange(1,5), # HAD2
                       np.arange(1,5), # HAD3
                       ))


class RpLayer(Layer):


  def __init__(self, rvec, **kwargs):
    super(RpLayer, self).__init__(**kwargs)
    self.rvec = rvec
    self.output_dim = (len(rvec),)



  def build( self, input_shape ):

    # Create the alpha trainable tf.variable
    self.__alpha = self.add_weight( name='alpha',
                               shape=(1,1),
                               initializer=tf.keras.initializers.RandomNormal(mean=1, stddev=0.5),
                               trainable=True)

    # Create the beta trainable tf.variable
    self.__beta = self.add_weight(name='beta',
                                  shape=(1,1),
                                  initializer=tf.keras.initializers.RandomNormal(mean=1, stddev=0.5),
                                  trainable=True)

    # Create the rvec tf.constant
    self.__rvec = K.constant( copy(self.rvec) )
    super(RpLayer, self).build(input_shape)



  def call(self, input):

    Ea = K.sign(input)*K.pow( K.abs(input), self.__alpha )
    rb =  K.pow(self.__rvec, self.__beta)
    Ea_sum = tf.reshape( K.sum( Ea, axis=1), (-1,1))
    out = (Ea*rb)/Ea_sum
    return out



  def get_output_shape_for(self, input_shape):
    return self.output_dim



