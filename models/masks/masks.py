import tensorflow as tf

def masks(enc_input, dec_input):
  enc_mask = _padding_mask(enc_input)
  latent_mask = _padding_mask(enc_input)

  dec_look_ahead = _look_ahead_mask(dec_input.shape[1])
  dec_padding_mask = _padding_mask(dec_input)
  dec_mask = tf.maximum(dec_look_ahead, dec_padding_mask)

  return enc_mask, latent_mask, dec_mask

def _padding_mask(x):
  x = tf.cast(tf.math.equal(x, 0), tf.float32)
  return x[:, tf.newaxis, tf.newaxis, :]

def _look_ahead_mask(seq_length):
  mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
  return mask
