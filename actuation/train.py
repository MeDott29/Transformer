import os
import argparse
import logging
import tensorflow as tf
from tqdm import tqdm
# Import losses:
from tensorflow.keras.losses import SparseCategoricalCrossentropy
# Import optimizers:
from tensorflow.keras.optimizers import Adam
# Import metrics:
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
# Import models:
from models.transformer import Transformer 
# Import processing:
from preprocess.process import Process
# Import masks:
from models.masks.masks import masks

class Train(object):
    def __init__(self, params):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.lr = params.lr
        self.epochs = params.epochs
        
        # Define loss:
        self.loss_object = SparseCategoricalCrossentropy()
        # Define optimizer:
        self.optimizer = Adam(learning_rate=self.lr)
        # Define metrics:
        self.train_loss = Mean(name='train_loss')
        self.train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = Mean(name='test_loss')
        self.test_accuracy = SparseCategoricalAccuracy(name='test_accuracy')
        
        # Define pre processor
        self.logger.info("Initializing data preprocessor...")
        preprocessor = Process(params.batch_size, params.pre_fetch)
        self.train_ds, self.test_ds, encoder_stats = preprocessor.get_datasets()
        
        # Define model dims
        d_model = 512
        ff_dim = 2048
        heads = 8
        encoder_dim = 6
        decoder_dim = 6
        dk = d_model // heads  # Changed from division to integer division
        dv = d_model // heads  # Changed from division to integer division
        vocab_size = encoder_stats['vocab_size']
        max_pos = 10000
        
        # Define model:
        self.logger.info("Initializing transformer model...")
        self.model = Transformer(d_model, ff_dim, dk, dv, heads,
            encoder_dim, decoder_dim, vocab_size, max_pos)
        
        # Define Checkpoints:
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimizer,
            net=self.model
        )
        
        # Create checkpoints directory if it doesn't exist
        ckpt_dir = f'checkpoints{params.ckpt_dir}'
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Define Checkpoint manager:
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, 
            ckpt_dir,
            max_to_keep=3
        )

    @tf.function
    def _update(self, inputs, labels):
        dec_inputs = labels[:, :-1]  # Fixed variable name
        dec_labels = labels[:, 1:]   # Renamed for clarity
        
        # Get masks
        inp_mask, latent_mask, dec_mask = masks(inputs, dec_inputs)
        
        with tf.GradientTape() as tape:
            predictions = self.model(
                inputs, 
                dec_inputs,
                inp_mask, 
                latent_mask, 
                dec_mask, 
                True
            )
            loss = self.loss_object(dec_labels, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(dec_labels, predictions)
        
        return loss

    @tf.function
    def _test(self, inputs, labels):
        dec_inputs = labels[:, :-1]
        dec_labels = labels[:, 1:]
        
        # Get masks
        inp_mask, latent_mask, dec_mask = masks(inputs, dec_inputs)
        
        predictions = self.model(
            inputs,
            dec_inputs,
            inp_mask,
            latent_mask,
            dec_mask,
            False
        )
        loss = self.loss_object(dec_labels, predictions)

        self.test_loss(loss)
        self.test_accuracy(dec_labels, predictions)
        
        return loss

    def _log_metrics(self, epoch):
        template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'
        self.logger.info(
            template.format(
                epoch + 1,
                self.train_loss.result(),
                self.train_accuracy.result() * 100,
                self.test_loss.result(),
                self.test_accuracy.result() * 100
            )
        )

    def _save(self):
        save_path = self.ckpt_manager.save()
        self.logger.info(f"Saved checkpoint for step {int(self.ckpt.step)}: {save_path}")

    def _restore(self):
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            self.logger.info(f"Restored from {self.ckpt_manager.latest_checkpoint}")
        else:
            self.logger.info("Initializing from scratch.")

    def _reset_metrics(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

    def train(self):
        self._restore()
        
        for epoch in range(self.epochs):
            self.logger.info(f"\nStarting epoch {epoch + 1}")
            
            # Training loop with progress bar
            train_batches = list(self.train_ds)
            with tqdm(total=len(train_batches), desc=f"Training epoch {epoch + 1}") as pbar:
                for batch, (inputs, labels) in enumerate(train_batches):
                    loss = self._update(inputs, labels)
                    pbar.set_postfix({
                        'loss': f'{loss.numpy():.4f}',
                        'accuracy': f'{self.train_accuracy.result().numpy():.2f}%'
                    })
                    pbar.update(1)
            
            # Testing loop with progress bar
            test_batches = list(self.test_ds)
            with tqdm(total=len(test_batches), desc=f"Testing epoch {epoch + 1}") as pbar:
                for inputs, labels in test_batches:
                    loss = self._test(inputs, labels)
                    pbar.set_postfix({
                        'loss': f'{loss.numpy():.4f}',
                        'accuracy': f'{self.test_accuracy.result().numpy():.2f}%'
                    })
                    pbar.update(1)
            
            self._log_metrics(epoch)
            self._save()
            self._reset_metrics()
            
            # Update checkpoint step
            self.ckpt.step.assign_add(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the transformer model')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=10000, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--pre_fetch', default=1, type=int, help='Number of batches to prefetch')
    parser.add_argument('--ckpt_dir', default='0', type=str, help='Checkpoint directory suffix')
    
    args = parser.parse_args()
    
    # Create main checkpoint directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    trainer = Train(args)
    trainer.train()