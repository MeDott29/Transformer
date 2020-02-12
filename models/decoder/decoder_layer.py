import tensorflow as tf
from tensorflow.keras.layers import (
  Dense, LayerNormalization, Dropout, Layer
)
from models.attention.multi_head_attention import MultiHeadAttention

class DecoderLayer(Layer):

  def __init__(self):
    super(DecoderLayer, self).__init__(ff_dim, d_model, dk, dv, heads)
    
    self.multiHeadAttention_input = MultiHeadAttention(d_model, dk, dv, heads)
    self.multiHeadAttention_latent = MultiHeadAttention(d_model, dk, dv, heads)

    self.ff = Dense(ff_dim)
    self.o = Dense(d_model)

    self.layernorm1 = LayerNormalization()
    self.layernorm2 = LayerNormalization()
    self.layernorm3 = LayerNormalization()

    self.dropout1 = Dropout()
    self.dropout2 = Dropout()
    self.dropout3 = Dropout()


  def call(self, x, training=False):
    input_attention = self.multiHeadAttention_input(x)
    input_attention = self.dropout1(input_attention, training=training)
    res_norm_input_attention = self.layernorm1(x + input_attention)

    latent_attention = self.multiHeadAttention_latent(x)
    latent_attention = self.dropout2(latent_attention, training=training)
    res_norm_latent_attention = self.layernorm1(latent_attention + res_norm_latent_attention)

    ff_h = self.ff(res_norm_latent_attention)
    ff_o = self.o(ff_h)
    ff_o = self.dropout3(ff_o)
    res_norm_ff = self.layernorm3(res_norm_attention + ff_o)
    return res_norm_ff 
