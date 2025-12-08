

__all__ = ['Job_v1']


from sklearn.model_selection import *
from Gaugi import save
from neuralnet.layers.RpLayer import RpLayer
# Just to remove the keras dependence
import tensorflow as tf
model_from_json = tf.keras.models.model_from_json
import json


class Job_v1( object ):


  __version =  1

  def __init__( self ):
    self.__sorts  = []
    self.__inits  = []
    self.__models = []
    self.__id     = None
    self.__metadata = None

  def setSorts(self, v):
    if type(v) is int:
      self.__sorts = [v]
    else:
      self.__sorts = v


  def setInits(self, v):
    if type(v) is int:
      self.__inits = range(v)
    else:
      self.__inits = v


  def getSorts(self):
    return self.__sorts


  def getInits(self):
    return self.__inits


  def setMetadata( self, d):
    self.__metadata = d


  def getMetadata(self):
    return self.__metadata


  def setModels(self, models, id_models):
    self.__models = list()
    if type(models) is not list:
      models=[models]
    for idx, model in enumerate(models):
      self.__models.append( {'model':  json.loads(model.to_json()), 'weights': model.get_weights() , 'id_model': id_models[idx]} )


  def getModels(self):
    # Loop over all keras model
    models = []; id_models = []
    for d in self.__models:
      model = model_from_json( json.dumps(d['model'], separators=(',', ':')) , custom_objects={'RpLayer':RpLayer} )
      model.set_weights( d['weights'] )
      models.append( model )
      id_models.append( d['id_model'] )
    return models, id_models


  def setId( self, id ):
    self.__id = id


  def id(self):
    return self.__id


  def toRawObj(self):
    return {
              'metadata' : self.__metadata,
              'id'       : self.__id,
              'sorts'    : self.__sorts,
              'inits'    : self.__inits,
              'models'   : self.__models,
              '__version': self.__version
    }

  def fromRawObj(self, d):
    self.__metadata = d['metadata']
    self.__id = d['id']
    self.__sorts = d['sorts']
    self.__inits = d['inits']
    self.__models = d['models']
    return self

  def save(self, fname):
    d = self.toRawObj()
    save( d, fname)


 

