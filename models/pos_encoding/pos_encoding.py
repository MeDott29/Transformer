import tensorflow as tf


def pos_encoding(d_model, vocab_size, pos):
  sin_input = pos / tf.math.pow(pos, (2 * vocab_size) / d_model)
  encoding_frequency = tf.math.sin(sin_input)
  return encoding_frequency
