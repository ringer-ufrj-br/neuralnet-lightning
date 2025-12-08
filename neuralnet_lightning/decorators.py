
__all__ = ["Summary", "Reference", "LinearFit"]


from Gaugi import Logger, StatusCode
from Gaugi.macros import *

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve

import rootplotlib as rpl
from tensorflow.keras.models import Model, model_from_json

import numpy as np
from copy import copy
import collections
import time

def sp_func(pd, fa):
  return np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )


#
# Decorate the history dictionary after the training phase with some useful controll values
#
class Summary( Logger ):

  #
  # Constructor
  #
  def __init__( self , detailed_info=True):
    Logger.__init__(self)
    self.detailed_info=detailed_info

  #
  # Use this method to decorate the keras history in the end of the training
  #
  def decorate( self, history, context ):

    d = {}
    x_train, y_train, _ = context.getHandler("trnData")
    x_val, y_val, _     = context.getHandler("valData")
    model               = context.getHandler( "model" )


    # Get the number of events for each set (train/val). Can be used to approx the number of
    # passed events in pd/fa analysis. Use this to integrate values (approx)
    sgn_total = len( y_train[y_train==1] )
    bkg_total = len( y_train[y_train!=1] )
    sgn_total_val = len( y_val[y_val==1] )
    bkg_total_val = len( y_val[y_val!=1] )


    MSG_INFO( self, "Starting the train summary..." )

    y_pred = model.predict( x_train, batch_size = 1024, verbose=0 )
    y_pred_val = model.predict( x_val, batch_size = 1024, verbose=0 )

    # get vectors for operation mode (train+val)
    y_pred_operation = np.concatenate( (y_pred, y_pred_val), axis=0)
    y_operation = np.concatenate((y_train,y_val), axis=0)

    
    # No threshold is needed
    d['auc'] = roc_auc_score(y_train, y_pred)
    d['auc_val'] = roc_auc_score(y_val, y_pred_val)
    d['auc_op'] = roc_auc_score(y_operation, y_pred_operation)


    # No threshold is needed
    d['mse'] = mean_squared_error(y_train, y_pred)
    d['mse_val'] = mean_squared_error(y_val, y_pred_val)
    d['mse_op'] = mean_squared_error(y_operation, y_pred_operation)

    if self.detailed_info:
      d['rocs'] = {}
      d['hists'] = {}

    m_step = 1e-2
    m_bins = np.arange(min(y_train), max(y_train)+m_step, step=m_step)
    
    
    # Here, the threshold is variable and the best values will
    # be setted by the max sp value found in hte roc curve
    # Training
    fa, pd, thresholds = roc_curve(y_train, y_pred)
    sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
    knee = np.argmax(sp)
    threshold = thresholds[knee]

    if self.detailed_info:
      d['rocs']['roc'] = (pd, fa)
      #d['rocs']['predictions'] = (y_pred, y_train)
      d['hists']['trn_sgn'] = np.histogram(y_pred[y_train == 1], bins=m_bins)
      d['hists']['trn_bkg'] = np.histogram(y_pred[y_train != 1], bins=m_bins)


    MSG_INFO( self, "Train samples     : Prob. det (%1.4f), False Alarm (%1.4f), SP (%1.4f), AUC (%1.4f) and MSE (%1.4f)",
        pd[knee], fa[knee], sp[knee], d['auc'], d['mse'])


    d['max_sp_pd'] = ( pd[knee], int(pd[knee]*sgn_total), sgn_total)
    d['max_sp_fa'] = ( fa[knee], int(fa[knee]*bkg_total), bkg_total)
    d['max_sp']    = sp[knee]
    d['acc']       = accuracy_score(y_train,y_pred>threshold)

    # Validation
    fa, pd, thresholds = roc_curve(y_val, y_pred_val)
    sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
    knee = np.argmax(sp)
    threshold = thresholds[knee]

    if self.detailed_info:
      d['rocs']['roc_val'] = (pd, fa)
      #d['rocs']['predictions_val'] = (y_pred_val, y_val)
      d['hists']['val_sgn'] = np.histogram(y_pred_val[y_val == 1], bins=m_bins)
      d['hists']['val_bkg'] = np.histogram(y_pred_val[y_val != 1], bins=m_bins)

    MSG_INFO( self, "Validation Samples: Prob. det (%1.4f), False Alarm (%1.4f), SP (%1.4f), AUC (%1.4f) and MSE (%1.4f)",
        pd[knee], fa[knee], sp[knee], d['auc_val'], d['mse_val'])


    d['max_sp_pd_val'] = (pd[knee], int(pd[knee]*sgn_total_val), sgn_total_val)
    d['max_sp_fa_val'] = (fa[knee], int(fa[knee]*bkg_total_val), bkg_total_val)
    d['max_sp_val']    = sp[knee]
    d['acc_val']       = accuracy_score(y_val,y_pred_val>threshold)

    # Operation
    fa, pd, thresholds = roc_curve(y_operation, y_pred_operation)
    sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
    knee = np.argmax(sp)
    threshold = thresholds[knee]

    if self.detailed_info:
      d['rocs']['roc_op'] = (pd, fa)
      # We dont need to attach y_op and y_pred_op since the user can concatenate train and val to get this. Just to save storage.
      #d['rocs']['predictions_op'] = (y_pred_operation, y_operations)
      d['hists']['op_sgn'] = np.histogram(y_pred_operation[y_operation == 1], bins=m_bins)
      d['hists']['op_bkg'] = np.histogram(y_pred_operation[y_operation != 1], bins=m_bins)

    MSG_INFO( self, "Operation Samples : Prob. det (%1.4f), False Alarm (%1.4f), SP (%1.4f), AUC (%1.4f) and MSE (%1.4f)",
        pd[knee], fa[knee], sp[knee], d['auc_val'], d['mse_val'])

    d['threshold_op'] = threshold
    d['max_sp_pd_op'] = ( pd[knee], int( pd[knee]*(sgn_total+sgn_total_val)), (sgn_total+sgn_total_val))
    d['max_sp_fa_op'] = ( fa[knee], int( fa[knee]*(bkg_total+bkg_total_val)), (bkg_total+bkg_total_val))
    d['max_sp_op'] = sp[knee]
    d['acc_op']              = accuracy_score(y_operation,y_pred_operation>threshold)

    history['summary'] = d

    return StatusCode.SUCCESS


 



#
# Use this class to decorate the history with the reference values configured by the user 
#
class Reference( Logger ):

  #
  # Constructor
  #
  def __init__( self , refFile=None, targets=None):
    Logger.__init__(self)
    self.__references = collections.OrderedDict()

    # Set all references from the reference file and target list
    if refFile and targets:
      from neuralnet.core import ReferenceReader
      refObj = ReferenceReader().load(refFile)
      for ref in targets:
        pd = (refObj.getSgnPassed(ref[0]) , refObj.getSgnTotal(ref[0]))
        fa = (refObj.getBkgPassed(ref[0]) , refObj.getBkgTotal(ref[0]))
        self.add_reference( ref[0], ref[1], pd, fa )
 

  #
  # Add the reference value
  #
  def add_reference( self, key, reference, pd, fa ):
    pd = [pd[0]/float(pd[1]), pd[0],pd[1]]
    fa = [fa[0]/float(fa[1]), fa[0],fa[1]]
    MSG_INFO( self, '%s | %s(pd=%1.2f, fa=%1.2f, sp=%1.2f)', key, reference, pd[0]*100, fa[0]*100, sp_func(pd[0],fa[0])*100 )
    self.__references[key] = {'pd':pd, 'fa':fa, 'sp':sp_func(pd[0],fa[0]), 'reference' : reference}


  #
  # decorate the history after the training phase
  #
  def decorate( self, history, context ):
    

    model  = context.getHandler("model")
    imodel = context.getHandler("imodel")
    index  = context.getHandler("index")
    sort   = context.getHandler("sort" )
    init   = context.getHandler("init" )

    x_train, y_train, _ = context.getHandler("trnData")
    x_val , y_val   , _ = context.getHandler("valData")

    y_pred     = model.predict( x_train, batch_size = 1024, verbose=0 )
    y_pred_val = model.predict( x_val  , batch_size = 1024, verbose=0 )

    # get vectors for operation mode (train+val)
    y_pred_operation = np.concatenate( (y_pred, y_pred_val), axis=0)
    y_operation = np.concatenate((y_train,y_val), axis=0)


    train_total = len(y_train)
    val_total = len(y_val)

    # Here, the threshold is variable and the best values will
    # be setted by the max sp value found in hte roc curve
    # Training
    fa, pd, thresholds = roc_curve(y_train, y_pred)
    sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )

    # Validation
    fa_val, pd_val, thresholds_val = roc_curve(y_val, y_pred_val)
    sp_val = np.sqrt(  np.sqrt(pd_val*(1-fa_val)) * (0.5*(pd_val+(1-fa_val)))  )

    # Operation
    fa_op, pd_op, thresholds_op = roc_curve(y_operation, y_pred_operation)
    sp_op = np.sqrt(  np.sqrt(pd_op*(1-fa_op)) * (0.5*(pd_op+(1-fa_op)))  )


    history['reference'] = {}

    for key, ref in self.__references.items():
      d = self.calculate( y_train, y_val , y_operation, ref, pd, fa, sp, thresholds, pd_val, fa_val, sp_val, thresholds_val, pd_op,fa_op,sp_op,thresholds_op )
      MSG_INFO(self, "          : %s", key )
      MSG_INFO(self, "Reference : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", ref['pd'][0]*100, ref['fa'][0]*100, ref['sp']*100 )
      MSG_INFO(self, "Train     : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd'][0]*100, d['fa'][0]*100, d['sp']*100 )
      MSG_INFO(self, "Validation: [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd_val'][0]*100, d['fa_val'][0]*100, d['sp_val']*100 )
      MSG_INFO(self, "Operation : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd_op'][0]*100, d['fa_op'][0]*100, d['sp_op']*100 )
      history['reference'][key] = d




  #
  # Calculate sp, pd and fake given a reference
  # 
  def calculate( self, y_train, y_val , y_op, ref, pd,fa,sp,thresholds, pd_val,fa_val,sp_val,thresholds_val, pd_op,fa_op,sp_op,thresholds_op ):

    d = {}
    def closest( values , ref ):
      index = np.abs(values-ref)
      index = index.argmin()
      return values[index], index


    # Check the reference counts
    op_total = len(y_op[y_op==1])
    if ref['pd'][2] !=  op_total:
      ref['pd'][2] = op_total
      ref['pd'][1] = int(ref['pd'][0]*op_total)

    # Check the reference counts
    op_total = len(y_op[y_op!=1])
    if ref['fa'][2] !=  op_total:
      ref['fa'][2] = op_total
      ref['fa'][1] = int(ref['fa'][0]*op_total)


    d['pd_ref'] = ref['pd']
    d['fa_ref'] = ref['fa']
    d['sp_ref'] = ref['sp']
    d['reference'] = ref['reference']


    # Train
    _, index = closest( pd, ref['pd'][0] )
    train_total = len(y_train[y_train==1])
    d['pd'] = ( pd[index],  int(train_total*float(pd[index])),train_total)
    train_total = len(y_train[y_train!=1])
    d['fa'] = ( fa[index],  int(train_total*float(fa[index])),train_total)
    d['sp'] = sp_func(d['pd'][0], d['fa'][0])
    d['threshold'] = thresholds[index]


    # Validation
    _, index = closest( pd_val, ref['pd'][0] )
    val_total = len(y_val[y_val==1])
    d['pd_val'] = ( pd_val[index],  int(val_total*float(pd_val[index])),val_total)
    val_total = len(y_val[y_val!=1])
    d['fa_val'] = ( fa_val[index],  int(val_total*float(fa_val[index])),val_total)
    d['sp_val'] = sp_func(d['pd_val'][0], d['fa_val'][0])
    d['threshold_val'] = thresholds_val[index]


    # Train + Validation
    _, index = closest( pd_op, ref['pd'][0] )
    op_total = len(y_op[y_op==1])
    d['pd_op'] = ( pd_op[index],  int(op_total*float(pd_op[index])),op_total)
    op_total = len(y_op[y_op!=1])
    d['fa_op'] = ( fa_op[index],  int(op_total*float(fa_op[index])),op_total)
    d['sp_op'] = sp_func(d['pd_op'][0], d['fa_op'][0])
    d['threshold_op'] = thresholds_op[index]

    return d



class LinearFit( Logger ):

  def __init__( self , refFile, targets, 
                xbin_size = 0.05,
                ybin_size = 0.5, 
                xmin = None,
                xmax = None,
                ymin = 16, 
                ymax = 60,
                min_avgmu = 0,
                max_avgmu = 100,
                batch_size=1024, 
                xmin_percentage=1,
                xmax_percentage=99,
                false_alarm_limit = 0.5):

    Logger.__init__(self)
    self.__references = collections.OrderedDict()
    self.xbin_size = xbin_size
    self.ymin = ymin
    self.ymax = ymax
    self.ybin_size = ybin_size
    self.batch_size = batch_size
    self.min_avgmu = min_avgmu
    self.max_avgmu = max_avgmu
    self.xmin_percentage = xmin_percentage
    self.xmax_percentage = xmax_percentage
    self.xmin = xmin
    self.xmax = xmax
    self.false_alarm_limit = false_alarm_limit

    # Set all references from the reference file and target list
    from neuralnet.core import ReferenceReader
    refObj = ReferenceReader().load(refFile)
    for ref in targets:
      pd = (refObj.getSgnPassed(ref[0]) , refObj.getSgnTotal(ref[0]))
      fa = (refObj.getBkgPassed(ref[0]) , refObj.getBkgTotal(ref[0]))
      self.add_reference( ref[0], ref[1], pd, fa )


  #
  # Add the reference value
  #
  def add_reference( self, key, reference, pd, fa ):
    pd = [pd[0]/float(pd[1]), pd[0],pd[1]]
    fa = [fa[0]/float(fa[1]), fa[0],fa[1]]
    MSG_INFO( self, '%s | %s(pd=%1.2f, fa=%1.2f, sp=%1.2f)', key, reference, pd[0]*100, fa[0]*100, sp_func(pd[0],fa[0])*100 )
    self.__references[key] = {'pd':pd, 'fa':fa, 'sp':sp_func(pd[0],fa[0]), 'reference' : reference}



  def decorate(self, history, context):

    import pandas as pd
    model  = context.getHandler("model")
    imodel = context.getHandler("imodel")
    index  = context.getHandler("index")
    sort   = context.getHandler("sort" )
    init   = context.getHandler("init" )

    history['fitting'] = {}
    x_train , y_train, avgmu_train = context.getHandler("trnData")
    x_val   , y_val  , avgmu_val   = context.getHandler("valData")


    model = Model(model.inputs, model.layers[-2].output)

    y_pred_train = model.predict( x_train, batch_size = 1024, verbose=1 ).flatten()
    y_pred_val   = model.predict( x_val  , batch_size = 1024, verbose=1 ).flatten()


    df_train = pd.DataFrame({ 'output' : y_pred_train, 
                              'avgmu'  : avgmu_train,
                              'target' : y_train})     
    df_val   = pd.DataFrame({ 'output' : y_pred_val, 
                              'avgmu'  : avgmu_val,
                              'target' : y_val})     


    # prepare all histograms
    xmin = int(np.percentile(df_train['output'].values , self.xmin_percentage)) if not self.xmin else self.xmin
    xmax = int(np.percentile(df_train['output'].values , self.xmax_percentage)) if not self.xmax else self.xmax
    
    if xmin == xmax:
      print('xmin == xmax -> \n make xmin = xmax -1 ')
      xmin = xmax - 1

    
    #print ('xmin = %1.2f, xmax = %1.2f'%(xmin,xmax))
    xbins = int((xmax-xmin)/self.xbin_size)
    ybins = int((self.ymax-self.ymin)/self.ybin_size)

    #
    # Fill train set
    #


    hist_signal = rpl.hist2d.make_hist('signal', 
                                        df_train.loc[df_train.target==1].output.values, 
                                        df_train.loc[df_train.target==1].avgmu.values , 
                                        xbins, xmin, xmax, ybins, self.ymin, self.ymax)
                                        
    hist_background = rpl.hist2d.make_hist('background',
                                           df_train.loc[df_train.target!=1].output.values, 
                                           df_train.loc[df_train.target!=1].avgmu.values, 
                                           xbins, xmin, xmax, ybins, self.ymin, self.ymax)


    # update avgmu values
    df_train[df_train.avgmu < self.min_avgmu] = self.min_avgmu
    df_train[df_train.avgmu > self.max_avgmu] = self.max_avgmu
    df_val[df_val.avgmu < self.min_avgmu] = self.min_avgmu
    df_val[df_val.avgmu > self.max_avgmu] = self.max_avgmu
    


    # Loop over all operation points
    for op_name, ref in self.__references.items():
      
      d = {}

      pd_ref = ref['pd'][0]
      pd_ref_passed = ref['pd'][1]
      pd_ref_total = ref['pd'][2]
      
      fa_ref = ref['fa'][0]
      fa_ref_passed = ref['fa'][1]
      fa_ref_total = ref['fa'][2]

      sp_ref = ref['sp']

      #
      # Train events (calculate slope and offset)
      #


      # calculate slope and offset from the train set
      slope, offset, converged = self.calculate( hist_signal, hist_background, pd_ref, false_alarm_limit=self.false_alarm_limit )
   
      df_train['dec'] = np.greater(df_train.output, slope*df_train.avgmu + offset)

      # signal (train)
      pd_train_passed = df_train.loc[(df_train.target==1) & (df_train.dec==True)].shape[0]
      pd_train_total  = df_train.loc[(df_train.target==1)].shape[0]
      pd_train        = pd_train_passed/pd_train_total

      # background (train)
      fa_train_passed = df_train.loc[(df_train.target!=1) & (df_train.dec==True)].shape[0]
      fa_train_total  = df_train.loc[(df_train.target!=1)].shape[0]
      fa_train = fa_train_passed/fa_train_total
      sp_train = np.sqrt(  np.sqrt(pd_train*(1-fa_train)) * (0.5*(pd_train+(1-fa_train)))  )


      #
      # Validation
      #

      df_val['dec'] = np.greater(df_val.output.values, slope*df_val.avgmu.values + offset)
      # signal (val)
      pd_val_passed = df_val.loc[(df_val.target==1) & (df_val.dec==True)].shape[0]
      pd_val_total  = df_val.loc[(df_val.target==1)].shape[0]
      pd_val = pd_val_passed/pd_val_total

      # background (val)
      fa_val_passed = df_val.loc[(df_val.target!=1) & (df_val.dec==True)].shape[0]
      fa_val_total  = df_val.loc[(df_val.target!=1)].shape[0]
      fa_val = fa_val_passed/fa_val_total
      sp_val = np.sqrt(  np.sqrt(pd_val*(1-fa_val)) * (0.5*(pd_val+(1-fa_val)))  )

      #
      # Operation (Train + val)
      #

      df_op = pd.concat([df_train, df_val], axis=0)

      df_op['dec'] = np.greater(df_op.output, slope*df_op.avgmu + offset)
      # signal (train)
      pd_op_passed = df_op.loc[(df_op.target==1) & (df_op.dec==True)].shape[0]
      pd_op_total  = df_op.loc[(df_op.target==1)].shape[0]
      pd_op  = pd_op_passed/pd_op_total
      # background (train)
      fa_op_passed = df_op.loc[(df_op.target!=1) & (df_op.dec==True)].shape[0]
      fa_op_total  = df_op.loc[(df_op.target!=1)].shape[0]
      fa_op  = fa_op_passed/fa_op_total
      sp_op  = np.sqrt(  np.sqrt(pd_op*(1-fa_op)) * (0.5*(pd_op+(1-fa_op)))  )

      d = {
            'pd_ref'  :   (pd_ref, pd_ref_passed, pd_ref_total),
            'fa_ref'  :   (fa_ref, fa_ref_passed, fa_ref_total),
            'sp_ref'  :   sp_ref,
            'pd'      :   (pd_train, pd_train_passed, pd_train_total),
            'fa'      :   (fa_train, fa_train_passed, fa_train_total),
            'sp'      :   sp_train,
            'pd_val'  :   (pd_val, pd_val_passed, pd_val_total),
            'fa_val'  :   (fa_val, fa_val_passed, fa_val_total),
            'sp_val'  :   sp_val,
            'pd_op'   :   (pd_op, pd_op_passed, pd_op_total),
            'fa_op'   :   (fa_op, fa_op_passed, fa_op_total),
            'sp_op'   :   sp_op,
            'params'  : {
              'xmin'        : xmin,
              'xmax'        : xmax,
              'xbin_size'   : self.xbin_size,
              'ymin'        : self.ymin,
              'ymax'        : self.ymax,
              'ybin_size'   : self.ybin_size,
              'min_avgmu'   : self.min_avgmu,
              'max_avgmu'   : self.max_avgmu,
              'slope'       : slope,
              'offset'      : offset,
              'converged'   : converged,
            }
      }

      MSG_INFO(self, "          : %s", op_name )
      MSG_INFO(self, "Reference : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", ref['pd'][0]*100, ref['fa'][0]*100, ref['sp']*100 )
      MSG_INFO(self, "Train     : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd'][0]*100, d['fa'][0]*100, d['sp']*100 )
      MSG_INFO(self, "Validation: [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd_val'][0]*100, d['fa_val'][0]*100, d['sp_val']*100 )
      MSG_INFO(self, "Operation : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd_op'][0]*100, d['fa_op'][0]*100, d['sp_op']*100 )

      history['fitting'][op_name] = d


  #
  # Fill correction table
  #
  def calculate( self, hist_signal, hist_background, pd_ref, false_alarm_limit = 0.5):
      
      false_alarm = 1.0
      while false_alarm > false_alarm_limit:
          # Get the threshold when we not apply any linear correction
          threshold, _ = self.find_threshold( hist_signal.ProjectionX(), pd_ref )
          # Get the efficiency without linear adjustment
          #signal_eff, signal_num, signal_den = self.calculate_num_and_den_from_hist(hist_signal, 0.0, threshold)
          # Apply the linear adjustment and fix it in case of positive slope
          slope, offset, status = self.fit( hist_signal, pd_ref )
          if slope > 0:
              slope = 0; offset = threshold
          # Get the efficiency with linear adjustment
          #signal_corrected_eff, signal_corrected_num, signal_corrected_den = self.calculate_num_and_den_from_hist(hist_signal, slope, offset)
          false_alarm, background_corrected_num, background_corrected_den = self.calculate_num_and_den_from_hist(hist_background, slope, offset)

          if false_alarm > false_alarm_limit:
              # Reduce the reference value by hand
              pd_ref-=0.0025

      return slope, offset, status


  #
  # Calculate the linear fit given a 2D histogram and reference value and return the slope and offset
  #
  def fit(self, th2, effref):
      x, y, errors = self.get_points(th2, effref )
      import array
      from ROOT import TGraphErrors, TF1
      g = TGraphErrors( len(x)
                           , array.array('d',y)
                           , array.array('d',x)
                           , array.array('d',[0.]*len(x))
                           , array.array('d',errors) )
      firstBinVal = th2.GetYaxis().GetBinLowEdge(th2.GetYaxis().GetFirst())
      lastBinVal = th2.GetYaxis().GetBinLowEdge(th2.GetYaxis().GetLast()+1)
      f1 = TF1('f1','pol1',firstBinVal, lastBinVal)
      r = g.Fit(f1,"FRSq")
      status = True if int(r)==0 else False

      slope = f1.GetParameter(1)
      offset = f1.GetParameter(0)
      return slope, offset, status


  #
  # Get all points in the 2D histogram given a reference value
  #
  def get_points( self, th2 , effref):
      nbinsy = th2.GetNbinsY()
      x = list(); y = list(); errors = list()
      for by in range(nbinsy):
          xproj = th2.ProjectionX('xproj'+str(time.time()),by+1,by+1)
          discr, error = self.find_threshold(xproj,effref)
          dbin = xproj.FindBin(discr)
          x.append(discr); y.append(th2.GetYaxis().GetBinCenter(by+1))
          errors.append( error )
      return x,y,errors


  #
  # Calculate the numerator and denomitator given the 2D histogram and slope/offset parameters
  #
  def calculate_num_and_den_from_hist(self, th2, slope, offset) :
    nbinsy = th2.GetNbinsY()
    th1_num = th2.ProjectionY(th2.GetName()+'_proj'+str(time.time()),1,1)
    th1_num.Reset("ICESM")
    numerator=0; denominator=0
    # Calculate how many events passed by the threshold
    for by in range(nbinsy) :
        xproj = th2.ProjectionX('xproj'+str(time.time()),by+1,by+1)
        # Apply the correction using ax+b formula
        threshold = slope*th2.GetYaxis().GetBinCenter(by+1)+ offset
        dbin = xproj.FindBin(threshold)
        num = xproj.Integral(dbin+1,xproj.GetNbinsX()+1)
        th1_num.SetBinContent(by+1,num)
        numerator+=num
        denominator+=xproj.Integral(-1, xproj.GetNbinsX()+1)
    return numerator/denominator if denominator!=0 else 0, numerator, denominator


  def calculate_num_and_den_from_df(self, df,  slope, offset) :
    df['dec'] = np.greater(df.output.values, slope*df.avgmu.values + offset)
    den = df.loc[(df.dec==True)].shape[0]
    num = df.shape[0]
    return den/num if den!=0 else 0, den, num


  #
  # Find the threshold given a reference value
  #
  def find_threshold(self, th1,effref):
    nbins = th1.GetNbinsX()
    fullArea = th1.Integral(0,nbins+1)
    if fullArea == 0:
        return 0,1
    notDetected = 0.0; i = 0
    while (1. - notDetected > effref):
        cutArea = th1.Integral(0,i)
        i+=1
        prevNotDetected = notDetected
        notDetected = cutArea/fullArea
    eff = 1. - notDetected
    prevEff = 1. -prevNotDetected
    deltaEff = (eff - prevEff)
    threshold = th1.GetBinCenter(i-1)+(effref-prevEff)/deltaEff*(th1.GetBinCenter(i)-th1.GetBinCenter(i-1))
    error = 1./np.sqrt(fullArea)
    return threshold, error







