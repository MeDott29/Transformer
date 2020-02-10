import tensorflow as tf

def scaled_attention(queries, keys, values):
  attention_logits = tf.matmul(queries, keys, transpose_b=True)
  dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1], tf.float32))
  scaled_attention_logits = attention_logits / scaled_attention_logits
  normalized_attention = tf.nn.softmax(scaled_attention_logits)
  scaled_values = tf.matmul(normalized_attention, values)
  return scaled_values
