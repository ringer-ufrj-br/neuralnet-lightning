__all__ = ['Reference_v1']

from Gaugi import save

class Reference_v1( object ):

  __version =  1


  def __init__( self ):

    self.__sgnRef = {}
    self.__bkgRef = {}
    self.__etBins = None
    self.__etaBins = None
    self.__etBinIdx = None
    self.__etaBinIdx = None


  def setEtBins(self, etBins ):
    self.__etBins = etBins

  def setEtaBins( self, etaBins ):
    self.__etaBins = etaBins


  def setEtBinIdx(self, etBinIdx ):
    self.__etBinIdx = etBinIdx

  def setEtaBinIdx(self, etaBinIdx ):
    self.__etaBinIdx = etaBinIdx


  def getEtBiinIdx(self):
    return self.__etBinIdx

  def getEtaBiinIdx(self):
    return self.__etaBinIdx


  def addSgn( self, reference, branch, passed, total):
    self.__sgnRef[ reference ] = {'passed':passed, 'total':total, 'reference': branch}

  def addBkg( self, reference, branch, passed, total):
    self.__bkgRef[ reference ] = {'passed':passed, 'total':total, 'reference': branch}


  def getSgnPassed(self, reference):
    return self.__sgnRef[reference]['passed']

  def getSgnTotal(self, reference):
    return self.__sgnRef[reference]['total']


  def getBkgPassed(self, reference):
    return self.__bkgRef[reference]['passed']

  def getBkgTotal(self, reference):
    return self.__bkgRef[reference]['total']


  def toRawObj(self):
    return { 
          'sgnRef'    : self.__sgnRef,
          'bkgRef'    : self.__bkgRef,
          'etBins'    : self.__etBins,
          'etaBins'   : self.__etaBins,
          'etBinIdx'  : self.__etBinIdx,
          'etaBinIdx' : self.__etaBinIdx,
          '__version' : self.__version
          }

  def fromRawObj( self, d):
    self.__bkgRef = d['bkgRef'].tolist()# NOTE: This is a hack
    self.__sgnRef = d['sgnRef'].tolist()# NOTE: This is a hack
    self.__etBins = d['etBins']
    self.__etaBins= d['etaBins']
    self.__etaBinIdx = d['etaBinIdx']
    self.__etBinIdx = d['etBinIdx']
    self.__version  = d['__version']
    return self


  def save(self, ofile):
    d = self.toRawObj()
    save( d, ofile)







