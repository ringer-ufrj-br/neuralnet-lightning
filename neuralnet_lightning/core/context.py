
__all__ = ["Context"]

from Gaugi import Logger
from Gaugi.macros import *
import collections

class Context(Logger):

  def __init__(self):
    Logger.__init__(self) 
    self.__containers = collections.OrderedDict()

  def setHandler(self, key, obj):
    if key in self.__containers.keys():
      MSG_ERROR(self, "Key %s exist into the event context. Attach is not possible...",key)
    else:
      self.__containers[key]=obj

  def getHandler(self,key):
    return None if not key in self.__containers.keys() else self.__containers[key]

  def clear(self):
    self.__containers = collections.OrderedDict()


