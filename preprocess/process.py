import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np
import os

class Process(object):
    def __init__(self, batch_size, pre_fetch):
        self.batch_size = batch_size
        self.pre_fetch = pre_fetch
        self.train_len = 392703
        self.vocab_size = 2**13  # Vocabulary size for tokenizer

    def get_datasets(self):
        # Load dataset with newer version string
        dataset, info = tfds.load('multi_nli', with_info=True)
        train_dataset = dataset['train']
        val_dataset = dataset['validation_matched']
        
        # Map the unpacking function
        train_dataset = train_dataset.map(self._unpack_vars)
        val_dataset = val_dataset.map(self._unpack_vars)
        
        # Build tokenizers
        self._build_encoders(train_dataset, val_dataset)
        
        # Take sample for testing/development
        train_dataset = train_dataset.take(100)
        val_dataset = val_dataset.take(100)
        
        # Encode datasets
        train_dataset = train_dataset.map(
            lambda inp, val: self._py_encode(inp, val, True))
        val_dataset = val_dataset.map(
            lambda inp, val: self._py_encode(inp, val, False))
        
        # Prepare final datasets
        train_dataset = self._prepare(train_dataset, True)
        val_dataset = self._prepare(val_dataset, False)
        
        return train_dataset, val_dataset, {'vocab_size': self.vocab_size}

    def _build_encoders(self, train, val):
        # Using BertTokenizer as a modern replacement for SubwordTextEncoder
        train_tokenizer_path = 'tokenizers/train_tokenizer'
        test_tokenizer_path = 'tokenizers/test_tokenizer'
        
        if os.path.exists(f'{train_tokenizer_path}/vocab.txt'):
            self.train_tokenizer = tf_text.BertTokenizer(
                f'{train_tokenizer_path}/vocab.txt')
        else:
            # Create vocabulary from training data
            vocab_set = set()
            for inp, _ in train:
                words = tf.strings.split(inp).numpy().flatten()
                vocab_set.update(words)
            
            vocabulary = sorted(list(vocab_set))
            os.makedirs(train_tokenizer_path, exist_ok=True)
            with open(f'{train_tokenizer_path}/vocab.txt', 'w') as f:
                for word in vocabulary:
                    f.write(f'{word.decode("utf-8")}\n')
            
            self.train_tokenizer = tf_text.BertTokenizer(
                f'{train_tokenizer_path}/vocab.txt')
        
        # Do the same for test tokenizer
        if os.path.exists(f'{test_tokenizer_path}/vocab.txt'):
            self.test_tokenizer = tf_text.BertTokenizer(
                f'{test_tokenizer_path}/vocab.txt')
        else:
            vocab_set = set()
            for _, label in train:
                words = tf.strings.split(label).numpy().flatten()
                vocab_set.update(words)
            
            vocabulary = sorted(list(vocab_set))
            os.makedirs(test_tokenizer_path, exist_ok=True)
            with open(f'{test_tokenizer_path}/vocab.txt', 'w') as f:
                for word in vocabulary:
                    f.write(f'{word.decode("utf-8")}\n')
            
            self.test_tokenizer = tf_text.BertTokenizer(
                f'{test_tokenizer_path}/vocab.txt')

    def _unpack_vars(self, dataset):
        text = dataset['premise']
        label = dataset['hypothesis']
        return text, label

    def _py_encode(self, text, label, train):
        text, label = tf.py_function(
            self._encode, [text, label, train], [tf.int64, tf.int64])
        text.set_shape([None])
        label.set_shape([None])
        return text, label

    def _encode(self, text, label, train):
        tokenizer = self.train_tokenizer if train else self.test_tokenizer
        
        # Add special tokens and encode
        text_tokens = tokenizer.tokenize(text)
        text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        text = tf.concat([[self.vocab_size], text_ids, [self.vocab_size + 1]], 0)
        
        label_tokens = tokenizer.tokenize(label)
        label_ids = tokenizer.convert_tokens_to_ids(label_tokens)
        label = tf.concat([[self.vocab_size], label_ids, [self.vocab_size + 1]], 0)
        
        return text, label

    def _prepare(self, dataset, train):
        if train:
            dataset = dataset.cache()
            dataset = dataset.shuffle(
                self.train_len).padded_batch(
                    self.batch_size, 
                    padded_shapes=([None], [None]))
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.padded_batch(
                self.batch_size, 
                padded_shapes=([None], [None]))
        return dataset