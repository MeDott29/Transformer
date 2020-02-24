import tensorflow as tf

def input_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  return seq
