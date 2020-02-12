import tensorflow as tf
from tensorflow.keras.layers import Dense 
from tensorflow.keras import Model
from models.encoder.encoder import Encoder
from models.encoder.decoder import Decoder

#Defining network Below:
class Transformer(Model):

  def __init__(self, d_model, ff_dim, dk, dv, heads):

    super(UNet, self).__init__()

    self.encoder = Encoder()
    self.decoder = Decoder()

    self.w_out = Dense()


  def call(self, x, training=False):
    latent = self.encoder(x)

    decoder_logits = self.decoder(latent, x)
    transformed_logits = self.w_out(decoder_logits, activation='softmax')

    return x
