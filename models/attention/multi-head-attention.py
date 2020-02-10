import tensorflow as tf
from tensorflow.keras.layers import (
  Dense, Layer
)
from models.attention.scaled_attention import scaled_attention


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, dk, dv, heads):
    super(MultiHeadAttention, self).__init__()

    assert d_model % heads == 0

    self.d_model = d_model
    self.heads = heads

    self.head_layers = []
    for i in range(heads):
      head = []
      head.append(Dense(dk))
      head.append(Dense(dk))
      head.append(Dense(dv))
      
    self.wo = Dense(d_model)

  def call(self, queries, keys, values):
    
    attention_heads = []
    for attention_layer in self.head_layers:
      projected_queries = attention_layer[0](queries)
      projected_keys = attention_layer[1](keys)
      projected_values = attention_layer[2](values)
      attention_heads.append(scaled_attention(
        projected_queries, projected_keys, projected_values
      )

    attention_logits = tf.concat(attention_heads, -1)
    projected_attention = self.wo(attention_logits)

    return prjected_attention
