

__all__ = ['TunedData_v1']


from Gaugi import save
from sklearn.model_selection import *
from tensorflow.keras.models import model_from_json
import json





class TunedData_v1( object ):

  __version =  1

  def __init__( self):

    self.__tunedData = []


  def attach( self, id_model, sort, init, tag, model, history, metadata={} ):

    self.__tunedData.append({'imodel'   : id_model,
                            'sort'     : sort,
                            'init'     : init,
                            'history'  : history,
                            'sequence' : json.loads(model.to_json()),
                            'weights'  : model.get_weights() ,
                            'metadata' : metadata,
                           })



  def attach_ctx( self, context ,  metadata={}):
    self.__tunedData.append({'imodel'   : context.getHandler("imodel"),
                            'sort'     : context.getHandler("sort"),
                            'init'     : context.getHandler("init"),
                            'history'  : context.getHandler("history"),
                            'sequence' : json.loads(context.getHandler("model").to_json()),
                            'weights'  : context.getHandler("model").get_weights() ,
                            'metadata' : metadata,
                            'time'     : context.getHandler('time'),
                           })




  def merge( self, obj ):
    self.__tunedData.extend( obj.get_data() )

  def get_data(self):
    return self.__tunedData


  def toRawObj(self):
    return {
              'tunedData' : self.__tunedData,
              '__version'  : self.__version
    }

  def fromRawObj( self, d):
    self.__tunedData = d['tunedData']
    return self 

  def save(self, ofile):
    d = self.toRawObj()
    save( d, ofile)







