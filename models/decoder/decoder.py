import tensorflow as tf
from tensorflow.keras.layers import (
  Dense, LayerNormalization, Dropout, Layer, Embedding
)
from models.decoder.decoder_layer import DecoderLayer

class Decoder(Layer):

  def __init__(self, ff_dim, d_model, dk, dv, 
        heads, vocab_size, decoder_dim, pos_encodings):
    super(Decoder, self).__init__()

    self.pos_encodings = pos_encodings
    
    self.embedding = Embedding(vocab_size, d_model)

    self.decoder_stack = []
    for i in range(decoder_dim):
      self.decoder_stack.append(DecoderLayer(ff_dim, d_model, dk, dv, heads))

    self.dropout1 = Dropout(.1)

  def call(self, x, latent, latent_mask,
      mask, training=False):

    seq_length = x.shape[1]

    input_embedding = self.embedding(x)
    pos_encodings = self.pos_encodings[:, :seq_length, :]
    adjusted_input_embedding = input_embedding + pos_encodings

    encoding = self.dropout1(adjusted_input_embedding)

    for decoder_layer in self.decoder_stack:
      encoding = decoder_layer(encoding, latent, 
          latent_mask, mask)
    return encoding 
