import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, LayerNormalization, Dropout, Layer, Embedding
)
from models.encoder.encoder_layer import EncoderLayer

class Encoder(Layer):

  def __init__(self, ff_dim, d_model, dk, dv, heads, 
        encoder_dim, vocab_size, pos_encodings, dynamic=True):

    super(Encoder, self).__init__()
    
    self.embedding = Embedding(vocab_size, d_model)

    self.pos_encodings = pos_encodings

    self.encoder_stack = []
    for i in range(encoder_dim):
      self.encoder_stack.append(EncoderLayer(ff_dim, d_model, dk, dv, heads))

    self.dropout1 = Dropout(.1)

  def call(self, x, training=False):
    seq_length = x.shape[1]

    input_embeddings = self.embedding(x)
    pos_embedding = self.pos_encodings[:, :seq_length, :]
    adjusted_input_embeddings = input_embeddings + pos_embedding

    encoding = dropout1(adjusted_input_embeddings)

    for encoder_layer in self.encoder_stack:
      encoding = encoder_layer(encoding)

    return encoding 
