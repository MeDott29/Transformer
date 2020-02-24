import tensorflow as tf
from tensorflow.keras.layers import (
  Dense, LayerNormalization, Dropout, Layer
)
from models.decoder.decoder_layer import DecoderLayer

class Decoder(Layer):

  def __init__(self):
    super(Decoder, self).__init__(ff_dim, d_model, dk, dv, 
        heads, vocab_size, decoder_dim, pos_encodings)

    self.pos_embedding = pos_embedding
    
    self.embedding = Embedding(vocab_size, d_model)

    self.decoder_stack = []
    for i in range(decoder_dim):
      self.encoder_stack.apped(DecoderLayer(ff_dim, d_model, dk, dv, heads))

    self.dropout1 = Dropout()

  def call(self, x, training=False):
    seq_length = tf.shape(x)[1]

    input_embedding = self.embedding(x)
    pos_embedding = self.pos_embedding[:, seq_length:, :]
    adjusted_input_embedding = input_embedding + pos_embedding

    encoding = dropout1(adjusted_input_embedding)

    for decoder_layer in self.decoder_stack:
      encoding = decoder_layer(encoding)

    return encoding 
