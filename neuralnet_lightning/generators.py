
__all__ = ['PatternGenerator']


from Gaugi import Logger
from Gaugi.macros import *

#
# Pattern generator used to read the data inline during the training phase
#
class PatternGenerator( Logger ):

  def __init__(self, path, generator , **kw):
    Logger.__init__(self)
    self.__path = path
    self.__generator = generator
    self.__kw = kw


  def __call__(self, cv, sort):
    MSG_INFO(self, "Reading %s...", self.__path)
    return self.__generator(self.__path, cv, sort, **self.__kw)


