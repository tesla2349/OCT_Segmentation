from keras import backend as K
import tensorflow as tf
import numpy as np

def dice_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
    return tf.reshape(1 - numerator / denominator, (-1, 1, 1))
    
def focal_loss(y_true, y_pred):
	def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
		weight_a = alpha * (1 - y_pred) ** gamma * targets
		weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
		return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b
	y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
	logits = tf.log(y_pred / (1 - y_pred))
	loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=.25, gamma=2., y_pred=y_pred)
	return tf.reduce_mean(loss)
		
def focal_dice_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
