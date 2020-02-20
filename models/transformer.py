import tensorflow as tf
from tensorflow.keras.layers import Dense 
from tensorflow.keras import Model
from models.encoder.encoder import Encoder
from models.decoder.decoder import Decoder
from models.pos_encoding.pos_encoding import pos_encoding

#Defining network Below:
class Transformer(Model):

  def __init__(self, d_model, ff_dim, dk, dv, heads, 
      encoder_dim, decoder_dim, label_vocab_dim,
      input_vocab_dim, max_pose):

    super(UNet, self).__init__()

    pos_encodings = pos_encoding(max_pose, d_model)

    self.encoder = Encoder(ff_dim, d_model, dk, dv, 
        heads, encoder_dim, input_vocab_dim, pos_encodings)
    self.decoder = Decoder(ff_dim, d_model, dk, dv, 
        heads, decoder_dim, label_vocab_dim, pos_encodings)

    self.w_out = Dense(label_vocab_dim)


  def call(self, encoder_input, decoder_input):
    pos_encoder_input = pos_encoding(self.d_model, vocab_size, encoder_input)
    latent = self.encoder(pos_encoder_input)

    pos_decoder_input = pos_encoding(self.d_model, self.vocab_size, decoder_input)
    decoder_logits = self.decoder(latent, x)
    transformed_logits = self.w_out(decoder_logits, activation='softmax')

    return x
