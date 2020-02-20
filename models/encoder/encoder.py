import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, LayerNormalization, Dropout, Layer
)
from models.encoder.encoder_layer import EncoderLayer

class Encoder(Layer):

  def __init__(self):
    super(Encoder, self).__init__(ff_dim, d_model, dk, dv, heads, 
        encoder_dim, vocab_size, pos_encodings)
    
    self.embedding = Embedding(vocab_size, d_model)

    self.encoder_stack = []
    for i in range(encoder_dim):
      self.encoder_stack.apped(EncoderLayer(ff_dim, d_model, dk, dv, heads))

    self.dropout1 = Dropout()

  def call(self, x, training=False):
    seq_length = tf.shape(x)[1]

    input_embedding = self.embedding(x)
    pos_embedding = self.pos_embedding[:, :seq_length, :]
    adjusted_input_embeddings = input_embeddings + pos_embedding

    encoding = dropout1(adjusted_input_embeddings)

    for encoder_layer in self.encoder_stack:
      encoding = encoder_layer(encoding)

    return encoding 
