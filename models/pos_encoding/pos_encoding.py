import tensorflow as tf
import numpy as np

def gen_angles(pos, i, d_model):
  power = 2 * (i // 2) / d_model
  return pos / np.power(10000, power)

def gen_pose_encoding(pos, d_model):
  encodings = gen_angles(
      np.arange(pos)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
  )

  # Sin of even dim
  encodings[:, 0::2] = np.sin(encodings[:, 0::2])

  # Cos of odd dim
  encodings[:, 1::2] = np.cos(encodings[:, 1::2])

  # Format shape to be 3 dim like inputs
  encodings = encodings[np.newaxis, :]

  return encodings
