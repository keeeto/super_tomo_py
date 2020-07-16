from tensorflow.keras import backend as keras
import tensorflow as tf

def dice_coef(y_true, y_pred):
    ''' 
    Calculate the dice loss

    Args: 
         y_true: the labeled mask corresponding to an rgb image
         y_pred: the predicted mask of an rgb image
    Returns: 
         dice_coeff: A metric that accounts for precision and recall
                           on the scale from 0 - 1. The closer to 1, the
                           better.

    Citation (MIT License): https://github.com/jocicmarko/
                            ultrasound-nerve-segmentation/blob/
                            master/train.py
    '''
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    smooth = 1.0
    return (2.0*intersection+smooth) / (keras.sum(y_true_f)+
            keras.sum(y_pred_f) + smooth)

def weighted_cross_entropy(beta):
  '''
  Weighted binary crossentropy. Very useful for class imbalanced image segmentation

  Args:
     beta: the weighting factor (float). For an explanation on how this works see:
     https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
  '''
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 
                                1 - tf.keras.backend.epsilon())

      return tf.math.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, 
                                   labels=y_true, pos_weight=beta)

    return tf.math.reduce_mean(loss)

  return loss

def balanced_cross_entropy(beta):
  '''
  Balanced binary crossentropy. Very useful for class imbalanced image segmentation

  Args:
     beta: the weighting factor (float). For an explanation on how this works see:
     https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
  '''
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.math.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)

    return tf.math.reduce_mean(loss * (1 - beta))

  return loss

def root_mean_squared_error(y_true, y_pred):
  '''
  Root mean squared error loss

  Args:
      y_true: the ground truth value
      y_pred: the predicted value

  Returns:
      loss: a float

  '''
  return keras.sqrt(keras.mean(keras.square(y_pred - y_true)))
