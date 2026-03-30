

__all__ = ["reprocess"]

from Gaugi import Logger, StatusCode, expand_folders, mkdir_p, load, save
from Gaugi.macros import *

from pprint import pprint

#from neuralnet.layers.RpLayer import RpLayer
from neuralnet.core.readers.versions import TunedData_v1
from neuralnet.core import Context


# Just to remove the keras dependence
import tensorflow as tf
model_from_json = tf.keras.models.model_from_json
import json


class Reprocess( Logger ):

  def __init__(self):

    Logger.__init__(self)



  #
  # run job
  #
  def __call__( self , generator, tunedFile, outputfile, crossval, decorators):


    context = Context()

    MSG_INFO( self, "Opening file %s...", tunedFile )
    raw = load(tunedFile)

    tunedData = TunedData_v1()

    for idx, tuned in enumerate(raw['tunedData']):

      # force the context is empty for each iteration
      context.clear()


      sort = tuned['sort']
      init = tuned['init']
      imodel = tuned['imodel']
      history = tuned['history']

      # get the current kfold and train, val sets
      x_train, x_val, y_train, y_val, avgmu_train, avgmu_val, index_from_cv = self.pattern_g( generator, crossval, sort )

      # recover keras model
      model = model_from_json( json.dumps(tuned['sequence'], separators=(',', ':')) )#, custom_objects={'RpLayer':RpLayer} )
      model.set_weights( tuned['weights'] )


      # Should not be store
      context.setHandler( "valData" , (x_val, y_val, avgmu_val)         )
      context.setHandler( "trnData" , (x_train, y_train, avgmu_train)   )
      context.setHandler( "index"   , index_from_cv        )
      context.setHandler( "crossval", crossval             )


      # It will be store into the file
      context.setHandler( "model"   , model         )
      context.setHandler( "sort"    , sort          )
      context.setHandler( "init"    , init          )
      context.setHandler( "imodel"  , imodel        )
      context.setHandler( "time"    , tuned['time'] )
      context.setHandler( "history" , history       )


      for tool in decorators:
        #MSG_INFO( self, "Executing the pos processor %s", tool.name() )
        tool.decorate( history, context )

      tunedData.attach_ctx( context )


    try:
      MSG_INFO( self, "Saving file..." )
      tunedData.save( outputfile+'/'+ tunedFile.split('/')[-1] )
    except Exception as e:
      MSG_FATAL( self, "Its not possible to save the tuned data: %s" , e )


    return StatusCode.SUCCESS




  def pattern_g( self, generator, crossval, sort ):
    # If the index is not set, you muat run the cross validation Kfold to get the index
    # this generator must be implemented by the user
    return generator(crossval, sort)





reprocess = Reprocess()

