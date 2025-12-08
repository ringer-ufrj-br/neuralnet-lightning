
__all__ = [
    "auc",
    "f1_score",
    "categorical_sp_metric",
    "sp_metric",
    "pd_metric",
    "fa_metric",
    "sp",
    "pd",
    "fa",


]

import numpy as np
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K
import tensorflow as tf
tf.executing_eagerly()


def auc(y_true, y_pred, num_thresholds=2000):
    import tensorflow as tf
    auc = tf.metrics.auc(y_true, y_pred, num_thresholds=num_thresholds)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def f1_score(y_true, y_pred):
    import tensorflow as tf
    f1 = tf.contrib.metrics.f1_score(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return f1


class categorical_sp(AUC):
    def __init__(self, *args, **kwargs):
        if ["multi_label" in kwargs]:
            del kwargs["multi_label"]
        super().__init__(multi_label=True, *args, **kwargs)
        tf.config.run_functions_eagerly(True)

    def result(self):
        fa: tf.Tensor = self.false_positives / \
            (self.true_negatives + self.false_positives + K.epsilon())
        pd: tf.Tensor = self.true_positives / \
            (self.true_positives + self.false_negatives + K.epsilon())
        sp: tf.Tensor = tf.norm(
            K.sqrt(K.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))), axis=1)
        knee = K.argmax(sp)
        return sp[knee] / K.sqrt(K.variable(value=self._num_labels))


class sp(AUC):

    # This implementation works with Tensorflow backend tensors.
    # That way, calculations happen faster and results can be seen
    # while training, not only after each epoch
    def result(self):

        # Add K.epsilon() for forbiding division by zero
        fa = self.false_positives / \
            (self.true_negatives + self.false_positives + K.epsilon())
        pd = self.true_positives / \
            (self.true_positives + self.false_negatives + K.epsilon())

        sp = K.sqrt(K.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))
        knee = K.argmax(sp)
        return sp[knee]


class pd(AUC):

    # This implementation works with Tensorflow backend tensors.
    # That way, calculations happen faster and results can be seen
    # while training, not only after each epoch
    def result(self):

        # Add K.epsilon() for forbiding division by zero
        fa = self.false_positives / \
            (self.true_negatives + self.false_positives + K.epsilon())
        pd = self.true_positives / \
            (self.true_positives + self.false_negatives + K.epsilon())

        sp = K.sqrt(K.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))
        knee = K.argmax(sp)
        return pd[knee]


class fa(AUC):

    # This implementation works with Tensorflow backend tensors.
    # That way, calculations happen faster and results can be seen
    # while training, not only after each epoch
    def result(self):

        # Add K.epsilon() for forbiding division by zero
        fa = self.false_positives / \
            (self.true_negatives + self.false_positives + K.epsilon())
        pd = self.true_positives / \
            (self.true_positives + self.false_negatives + K.epsilon())

        sp = K.sqrt(K.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))
        knee = K.argmax(sp)
        return fa[knee]


categorical_sp_metric = categorical_sp(num_thresholds=1000,
                                       curve="ROC",
                                       summation_method="interpolation",
                                       name=None,
                                       dtype=None,
                                       thresholds=None,
                                       multi_label=True,
                                       label_weights=None,)

sp_metric = sp(num_thresholds=1000,
               curve="ROC",
               summation_method="interpolation",
               name=None,
               dtype=None,
               thresholds=None,
               multi_label=False,
               label_weights=None,)

pd_metric = pd(num_thresholds=1000,
               curve="ROC",
               summation_method="interpolation",
               name=None,
               dtype=None,
               thresholds=None,
               multi_label=False,
               label_weights=None,)

fa_metric = fa(num_thresholds=1000,
               curve="ROC",
               summation_method="interpolation",
               name=None,
               dtype=None,
               thresholds=None,
               multi_label=False,
               label_weights=None,)
