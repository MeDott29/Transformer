import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

class Process(object):

  def __init__(self, batch_size, pre_fetch):
    self.batch_size = batch_size
    self.pre_fetch = pre_fetch

  def get_datasets(self):
    dataset, info = tfds.load('multi_nli:1.0.0', with_info=True)
    train = dataset['train']
    test = dataset['validation_matched']
    return train_dataset, test_dataset

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
