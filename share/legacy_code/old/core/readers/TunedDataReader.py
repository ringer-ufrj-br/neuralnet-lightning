
__all__ = ['TunedDataReader']


from Gaugi import Logger
from Gaugi.macros import *
from Gaugi import load
from Gaugi import expand_folders, progressbar


class TunedDataReader( Logger ):

  def __init__( self, **kw ):
    Logger.__init__(self, kw)
    self._obj = None

  def load( self, fList ):


    fList = expand_folders(fList)
    fList = [f for f in fList if '.npz' in f]
    from neuralnet import TunedData_v1
    self._obj = TunedData_v1()
    for inputFile in progressbar(fList, "Reading tuned data collection..." ):

      raw = load( inputFile )
      # get the file version
      version = raw['__version']
      # the current file version
      if version == 1:
        obj = TunedData_v1().fromRawObj( raw )
        self._obj.merge( obj )
      else:
        # error because the file does not exist
        self._logger.fatal( 'File version (%d) not supported in (%s)', version, inputFile)

    # return the list of keras models
    return self._obj
    

  def save(self, obj, ofile):
    obj.save(ofile)


  def object(self):
    return self._obj






