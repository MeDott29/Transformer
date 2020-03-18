import tensorflow as tf
from tensorflow.keras.layers import Dense 
from tensorflow.keras import Model
from models.encoder.encoder import Encoder
from models.decoder.decoder import Decoder
from models.pos_encoding.pos_encoding import pos_encoding

#Defining network Below:
class Transformer(Model):

  def __init__(self, d_model, ff_dim, dk, dv, heads, 
      encoder_dim, decoder_dim, vocab_size, max_pose):

    super(Transformer, self).__init__()
    
    heads = 8
    d_model = 512
    max_pose = 512

    self.d_model = d_model
    self.vocab_size = vocab_size

    pos_encodings = pos_encoding(max_pose, d_model)

    self.encoder = Encoder(ff_dim, d_model, dk, dv, 
        heads, encoder_dim, vocab_size, pos_encodings)
    self.decoder = Decoder(ff_dim, d_model, dk, dv, 
        heads, decoder_dim, vocab_size, pos_encodings)

    self.w_out = Dense(vocab_size, activation='softmax')


  def call(self, encoder_input, decoder_input, training=False):
    latent = self.encoder(encoder_input)
    decoder_logits = self.decoder(decoder_input, latent)
    transformed_logits = self.w_out(decoder_logits)
    return transformed_logits
