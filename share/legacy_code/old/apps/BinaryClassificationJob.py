

__all__ = ['BinaryClassificationJob']

from Gaugi import Logger, StatusCode, declareProperty
from Gaugi.macros import *

import tensorflow as tf
from tensorflow.keras.models import clone_model
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight

from datetime import datetime
from copy import deepcopy, copy
import numpy as np
import pandas as pd




class BinaryClassificationJob( Logger ):

  def __init__(self , pattern_generator, crossval, **kw ):

    Logger.__init__(self)

    self.__pattern_generator = pattern_generator
    self.crossval = crossval


    declareProperty( self, kw, 'optimizer'      , 'adam'                )
    declareProperty( self, kw, 'loss'           , 'binary_crossentropy' )
    declareProperty( self, kw, 'epochs'         , 1000                  )
    declareProperty( self, kw, 'batch_size'     , 1024                  )
    declareProperty( self, kw, 'callbacks'      , []                    )
    declareProperty( self, kw, 'metrics'        , []                    )
    declareProperty( self, kw, 'sorts'          , range(1)              )
    declareProperty( self, kw, 'inits'          , 1                     )
    declareProperty( self, kw, 'decorators'     , []                    )
    declareProperty( self, kw, 'plots'          , []                    )
    declareProperty( self, kw, 'job'            , None                  )
    declareProperty( self, kw, 'model_generator', None                  )
    declareProperty( self, kw, 'verbose'        , True                  )
    declareProperty( self, kw, 'class_weight'   , True                  )
    declareProperty( self, kw, 'save_history'   , True                  )

    # Wandb application
    declareProperty( self, kw, 'use_wandb'         , False                  )
    declareProperty( self, kw, 'wandb_usernmame'   , 'jodafon'              )
    declareProperty( self, kw, 'wandb_task'        , 'user.jodafons.task'   )


    # read the job configuration from file
    MSG_INFO(self, 'Setup job...')
    if self.job:
      if type(self.job) is str:
        MSG_INFO( self, 'Reading job configuration from: %s', self.job )
        from neuralnet.core.readers import JobReader
        job = JobReader().load( self.job )
      else:
        job = self.job
      # retrive sort/init lists from file
      self.sorts = job.getSorts()
      self.inits = job.getInits()
      self.__models, self.__id_models = job.getModels()
      self.__jobId = job.id()


    # get model and tag from model file or lists
    declareProperty( self, kw, 'models', None )

    MSG_INFO(self, 'Setup models...')

    if self.models:
      self.__models = self.models
      self.__id_models = [id for id in range(len(self.models))]
      self.__jobId = 0



    declareProperty( self, kw, 'outputFile' , None )

    MSG_INFO(self, 'Setup output file...')

    if self.outputFile:
      from neuralnet.core.readers.versions import TunedData_v1
      self.__tunedData = TunedData_v1()


    from neuralnet import Context
    self.__context = Context()
    self.__index_from_cv = None
    self.__trained_models = []



  #
  # Sorts setter and getter
  #
  @property
  def sorts(self):
    return self.__sorts

  @sorts.setter
  def sorts( self, s):
    if type(s) is int:
      self.__sorts = range(s)
    else:
      self.__sorts = s


  #
  # Init setter and getter
  #
  @property
  def inits(self):
    return self.__inits

  @inits.setter
  def inits( self, s):
    if type(s) is int:
      self.__inits = range(s)
    else:
      self.__inits = s



  #
  # run job
  #
  def run( self ):

    MSG_INFO(self, 'Running....')
    tf.config.run_functions_eagerly(False)

    for isort, sort in enumerate( self.sorts ):

      # get the current kfold and train, val sets
      x_train, x_val, y_train, y_val, avgmu_train, avgmu_val, self._index_from_cv = self.pattern_g( self.__pattern_generator, self.crossval, sort )


      if ( len(np.unique(y_train))!= 2 ):
        MSG_FATAL(self, "The number of targets is different than 2. This job is used for binary classification.")


      # check if there are fewer events than the batch_size
      _, n_evt_per_class = np.unique(y_train, return_counts=True)
      batch_size = (self.batch_size if np.min(n_evt_per_class) > self.batch_size
                     else np.min(n_evt_per_class))

      MSG_INFO( self, "Using %d as batch size.", batch_size)

      for imodel, model in enumerate( self.__models ):

        for iinit, init in enumerate(self.inits):



          # force the context is empty for each training
          self.__context.clear()
          self.__context.setHandler( "jobId"    , self.__jobId         )
          self.__context.setHandler( "crossval" , self.crossval        )
          self.__context.setHandler( "index"    , self.__index_from_cv )
          self.__context.setHandler( "valData"  , (x_val  , y_val  , avgmu_val  )   )
          self.__context.setHandler( "trnData"  , (x_train, y_train, avgmu_train)   )
          #self.__context.setHandler( "features" , features             )


          # get the model "ptr" for this sort, init and model index
          if self.model_generator:
            MSG_INFO( self, "Apply model generator..." )
            model_for_this_init = self.model_generator( sort )
          else: 
            model_for_this_init = clone_model(model) # get only the model


          try:

            model_for_this_init.compile( self.optimizer,
                      loss = self.loss,
                      # protection for functions or classes with internal variables
                      # this copy avoid the current training effect the next one.
                      metrics = deepcopy(self.metrics),
                      #metrics = self.metrics,
                      )
            model_for_this_init.summary()
          except RuntimeError as e:
            MSG_FATAL( self, "Compilation model error: %s" , e)


          MSG_INFO( self, "Training model id (%d) using sort (%d) and init (%d)", self.__id_models[imodel], sort, init )
          MSG_INFO( self, "Train Samples      :  (%d, %d)", len(y_train[y_train==1]), len(y_train[y_train!=1]))
          MSG_INFO( self, "Validation Samples :  (%d, %d)", len(y_val[y_val==1]),len(y_val[y_val!=1]))


          callbacks = copy(self.callbacks)
          for callback in callbacks:
            if hasattr(callback, 'set_validation_data'):
              callback.set_validation_data( (x_val,y_val) )


          if self.use_wandb:
            from neuralnet.callbacks import wandb
            wb = wandb(self.wandb_username, self.wanddb_task)
            name = 'model_%d_sort_%s_init_%s' % (imodel, sort, iinit)
            wb.init(name)
            callbacks.append(wb)


          start = datetime.now()


          if self.class_weight:
            classes = np.unique(y_train).tolist()
            # [-1,1] or [0,1]
            weights = compute_class_weight('balanced',classes,y_train)
            class_weights = {cl : weights[idx] for idx, cl in enumerate(classes)}
            sample_weight = np.ones_like(y_train, dtype=np.float32)
            sample_weight[y_train==1] = weights[1]
            sample_weight[y_train!=1] = weights[0] 
            print(class_weights)
          else:
            sample_weight = np.ones_like(y_train)

    
          history = model_for_this_init.fit(x_train, y_train,
                              epochs          = self.epochs,
                              batch_size      = batch_size,
                              verbose         = self.verbose,
                              validation_data = (x_val,y_val),
                              # copy protection to avoid the interruption or interference
                              # in the next training (e.g: early stop)
                              # bugfix: https://stackoverflow.com/questions/63158424/why-does-keras-model-fit-with-sample-weight-have-long-initialization-time
                              sample_weight   = pd.Series(sample_weight),
                              callbacks       = callbacks,
                              shuffle         = True).history

          end = datetime.now()

          # be sure that model is already trained
          self.__context.setHandler( "model"   , model_for_this_init     )
          self.__context.setHandler( "sort"    , sort                    )
          self.__context.setHandler( "init"    , init                    )
          self.__context.setHandler( "imodel"  , self.__id_models[imodel])
          self.__context.setHandler( "time"    , end-start)


          if not self.save_history:
            # overwrite to slim version. This is used to reduce the output size
            history = {}

          self.__context.setHandler( "history", history )

          for tool in self.decorators:
            #MSG_INFO( self, "Executing the pos processor %s", tool.name() )
            tool.decorate( history, self.__context )

          for plot in self.plots:
            plot( self.__context )

          # add the tuned parameters to the output file
          if self.outputFile:
            self.__tunedData.attach_ctx( self.__context )

          # Clear everything for the next init
          K.clear_session()
          self.__trained_models.append( (model_for_this_init, history) )

      # You must clean everythin before reopen the dataset
      self.__context.clear()
      # Clear the keras once again just to be sure
      K.clear_session()


    # End of training
    try:
      # prepare to save the tuned data
      if self.outputFile:
        self.__tunedData.save( self.outputFile )
    except Exception as e:
      MSG_FATAL( self, "Its not possible to save the tuned data: %s" , e )


    return StatusCode.SUCCESS




  def pattern_g( self, generator, crossval, sort ):
    # If the index is not set, you muat run the cross validation Kfold to get the index
    # this generator must be implemented by the user
    return generator(crossval, sort)



  def getAllModels(self):
    return self.__trained_models


