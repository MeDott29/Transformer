import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, LayerNormalization, Dropout, Layer
)
from models.encoder.decoder_layer import DecoderLayer
from models.pos_encoding.pos_encoding import PosEncoding

class Decoder(Layer, ff_dim, d_model):

  def __init__(self):
    super(Decoder, self).__init__()
    
    self.posEncoding = PosEncoding()

    self.embedding = Embedding()

    self.decoder_stack = []
    for i in range(decoder_dim):
      self.encoder_stack.apped(DecoderLayer(ff_dim, d_model))

    self.dropout1 = Dropout()

  def call(self, x, training=False):
    input_embedding = self.embedding(x)
    pos_embedding = self.posEncoding(x)

    encoding = dropout1(pos_embedding)

    for decoder_layer in self.decoder_stack:
      encoding = decoder_layer(encoding)

    return encoding 
