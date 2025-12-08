__all__ = ["model_generator_base"]

import tensorflow as tf
from tensorflow.keras.models import clone_model, Model, model_from_json
from tensorflow.keras import layers
from neuralnet.core import TunedDataReader
from Gaugi.macros import *
from Gaugi import Logger
import json



class model_generator_base( Logger ):

  #
  # Constructor
  #
  def __init__( self ):
    Logger.__init__(self)


  #
  # Call method
  #
  def __call__( self, sort ):
    pass


  #
  # tranfer the source weights to the target layer
  #
  def transfer_weights( self, from_model, from_layer, to_model, to_layer, trainable=True ):

    source_layer = None
    # Loop over all layers in the source model
    for layer in from_model.layers:
      if layer.name == from_layer:
        source_layer = layer

    if not source_layer:
      MSG_FATAL( self, "From model with layer %s does not exist.", from_layer)

    # Loop of all layers in the target model
    for target_layer in to_model.layers:
      if target_layer.name == to_layer:
        MSG_INFO(self, "Copy weights from %s to %s", from_layer, to_layer)
        target_layer.set_weights( source_layer.get_weights() )
        target_layer.trainable = trainable



  #
  # Load all tuned filed
  #
  def load_models( self, path ):
    tunedData = TunedDataReader()
    tunedData.load( path )
    return tunedData.object().get_data()



  #
  # Get best model given the sort number
  #
  def get_best_model( self , tuned_list, sort, imodel):

    best_model=None; best_sp=-999
    # Loop over all tuned files
    for tuned in tuned_list:
      history = tuned['history']
      if tuned['sort']==sort and best_sp < history['summary']['max_sp_op'] and tuned['imodel']==imodel:
        # read the model
        best_model = model_from_json( json.dumps(tuned['sequence'], separators=(',', ':')) )
        best_model.set_weights( tuned['weights'] )
        best_model = Model(best_model.inputs, best_model.layers[-1].output)
        best_sp =history['summary']['max_sp_op']

    return best_model




