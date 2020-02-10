import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, LayerNormalization, Dropout
)
from tensorflow.keras import Model
from models.attention.multi_head_attention import MultiHeadAttention

class EncoderLayer(Model, ff_dim, d_model):
  def __init__(self):
    super(EncoderLayer, self).__init__()
    
    self.multiHeadAttention = MultiHeadAttention()

    self.ff = Dense(ff_dim)
    self.o = Dense(d_model)

    self.layernorm1 = LayerNormalization()
    self.layernorm2 = LayerNormalization()

    self.dropout1 = Dropout()
    self.dropout2 = Dropout()


  def call(self, x, training=False):
    attention = self.multiHeadAttention(x)
    attention = self.dropout1(attention, training=training)
    res_norm_attention = self.layernorm1(x + attention)

    ff_h = self.ff(res_norm_attention)
    ff_o = self.o(ff_h)
    ff_o = self.dropout2(ff_o)
    res_norm_ff = self.layernorm2(res_norm_attention + ff_o)
    return res_norm_ff 
