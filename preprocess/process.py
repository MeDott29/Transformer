import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import os

class Process(object):

  def __init__(self, batch_size, pre_fetch):
    self.batch_size = batch_size
    self.pre_fetch = pre_fetch

  def get_datasets(self):
    dataset, info = tfds.load('multi_nli:1.0.0', with_info=True)
    train_dataset = dataset['train']
    val_dataset = dataset['validation_matched']
    train_dataset = train_dataset.map(self._unpack_vars)
    val_dataset = val_dataset.map(self._unpack_vars)
    self._build_encoders(train_dataset, val_dataset)
    train_dataset = train_dataset.map(
        lambda inp, val: self._py_encode(inp, val, True))
    val_dataset = val_dataset.map(
        lambda inp, val: self._py_encode(inp, val, False))
    return train_dataset, val_dataset

  def _build_encoders(self, train, val):
    train_tokenizer = None
    if (os.path.exists('tokenizers/train_tokenizer.subwords')):
      train_tokenizer = tfds.features.text.SubwordTextEncoder \
        .load_from_file('tokenizers/train_tokenizer')
    else:
      train_tokenizer = tfds.features.text.SubwordTextEncoder \
        .build_from_corpus(
            (inp.numpy() for inp, label in train), 
            2**13)
      train_tokenizer.save_to_file('tokenizers/train_tokenizer')
  
    test_tokenizer = None
    if (os.path.exists('tokenizers/test_tokenizer.subwords')):
      test_tokenizer = tfds.features.text.SubwordTextEncoder \
        .load_from_file('tokenizers/test_tokenizer')
    else:
      test_tokenizer = tfds.features.text.SubwordTextEncoder \
        .build_from_corpus(
            (label.numpy() for inp, label in train), 2**13)
      test_tokenizer.save_to_file('tokenizers/test_tokenizer')

    self.train_tokenizer = train_tokenizer
    self.test_tokenizer = test_tokenizer

  def _unpack_vars(self, dataset):
    text = dataset['premise']
    label = dataset['hypothesis']

    return text, label

  def _py_encode(self, text, label, train):
    text, label = tf.py_function(
        self._encode, [text, label, train], [tf.int64, tf.int64])
    return text, label

  def _encode(self, text, label, train):
    tokenizer = None
    if train:
      tokenizer = self.train_tokenizer
    else:
      tokenizer = self.test_tokenizer
    text = [tokenizer.vocab_size] + tokenizer.encode(
        text.numpy()) + [tokenizer.vocab_size + 1]

    label = [tokenizer.vocab_size] + tokenizer.encode(
        label.numpy()) + [tokenizer.vocab_size + 1]

    return text, label

  def _build_dataset(self, data_dir, train_data):
    data_files = self._load_files(data_dir)
    dataset = self._load_dataset(data_files)

    if train_data:
      dataset = self._normalize_data(dataset)
      dataset = self._augment_data(dataset)

    dataset = self._prepare(dataset)
    return dataset 

  def _prepare(self, dataset, train):
    if train:
      dataset = dataset.cache().shuffle(data_len).batch(self.batch_size)
      dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
      dataset = dataset.batch(self.batch_size)
    return dataset  

 # def _normalize(self, input_, label):
 #   # Normalize input data: 

 #   return input_, label
 # 
 # def _augment_data(self, input_, label):
 #   # Augment training data here

 #   return input_, label
  
  @tf.function
  def _load_dataset(self, data_files):
    dataset = tf.data.Dataset.from_tensor_slices(data_files)
    dataset = dataset.map(
      self._load_datapoint, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset 
  
  @tf.function
  def _load_datapoint(self, input_file, label_file):
    # Add code to load input and label file:

    return input_, label
  
#  def _test(self):
    # Add code to display or test data in some way

p = Process(10,1)
train, val = p.get_datasets()
for sent, hyp in train.take(1):
  print(sent.numpy())
  print(hyp.numpy())
for sent, hyp in val.take(1):
  print(sent.numpy())
  print(hyp.numpy())
