import tensorflow as tf
import numpy as np
import os
import logging
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import your model (assuming it's in a separate file)
# from enhanced_minigpt import EnhancedMiniGPT, ModelConfig, MiniGPTTrainer

def create_dummy_dataset(vocab_size=50257, seq_len=512, batch_size=32, num_samples=1000):
    """Create a dummy dataset for testing"""
    def data_generator():
        for _ in range(num_samples):
            # Generate random sequences
            sequence = np.random.randint(0, vocab_size, size=seq_len, dtype=np.int32)
            yield {'input_ids': sequence}
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature={
            'input_ids': tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
        }
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_text_dataset(text_file_path, tokenizer, seq_len=512, batch_size=32):
    """Create dataset from text file"""
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize the entire text
        tokens = tokenizer.encode(text)
        
        # Create sequences
        sequences = []
        for i in range(0, len(tokens) - seq_len, seq_len // 2):  # 50% overlap
            sequences.append(tokens[i:i + seq_len])
        
        def data_generator():
            for seq in sequences:
                yield {'input_ids': np.array(seq, dtype=np.int32)}
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature={
                'input_ids': tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
            }
        )
        
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    except Exception as e:
        logger.warning(f"Failed to create text dataset: {e}")
        return None

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    """Custom checkpoint callback that saves both weights and full model"""
    
    def __init__(self, filepath, save_weights_only=False, save_best_only=False, 
                 monitor='loss', mode='min', save_freq='epoch', verbose=1):
        super().__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_freq = save_freq
        self.verbose = verbose
        
        if mode == 'min':
            self.best = float('inf')
            self.monitor_op = np.less
        else:
            self.best = -float('inf')
            self.monitor_op = np.greater
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Format filepath
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        
        # Check if we should save
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose > 0:
                logger.warning(f"Can't find {self.monitor} in logs")
            return
        
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
            else:
                return
        
        try:
            if self.save_weights_only:
                # Ensure .weights.h5 extension
                if not filepath.endswith('.weights.h5'):
                    filepath = filepath + '.weights.h5'
                self.model.save_weights(filepath)
            else:
                # Save full model
                if not filepath.endswith('.keras'):
                    filepath = filepath + '.keras'
                self.model.save(filepath)
            
            if self.verbose > 0:
                logger.info(f"Saved model at epoch {epoch + 1}: {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

def setup_callbacks(checkpoint_dir="./checkpoints", monitor='loss'):
    """Setup training callbacks"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        # Custom checkpoint callback
        CustomModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "minigpt_epoch_{epoch:02d}_loss_{loss:.4f}"),
            save_weights_only=True,
            save_best_only=True,
            monitor=monitor,
            mode='min',
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(checkpoint_dir, "logs"),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # CSV logger
        tf.keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, "training_log.csv"),
            append=True
        )
    ]
    
    return callbacks

def train_model():
    """Main training function"""
    try:
        # Import your model classes here
        from enhanced_minigpt import EnhancedMiniGPT, ModelConfig
        
        logger.info("Starting MiniGPT training")
        
        # Configuration
        config = ModelConfig(
            vocab_size=50257,
            max_seq_len=512,
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            ffn_dim=3072,
            dropout=0.1,
            use_custom_attention=True
        )
        
        # Create model
        logger.info("Creating model...")
        model = EnhancedMiniGPT(config)
        
        # Build model
        model.build_model()
        logger.info(f"Model created with {model.count_params():,} parameters")
        
        # Setup optimizer
        initial_learning_rate = 1e-4
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=10000,
            alpha=0.1
        )
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.95,
            epsilon=1e-8
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Create dataset
        logger.info("Creating dataset...")
        
        # Try to load from text file first, fallback to dummy data
        text_file = "train_data.txt"  # Change this to your text file path
        dataset = None
        
        if model.tokenizer and os.path.exists(text_file):
            dataset = create_text_dataset(text_file, model.tokenizer, 
                                        seq_len=config.max_seq_len, batch_size=8)
            logger.info(f"Loaded dataset from {text_file}")
        
        if dataset is None:
            logger.info("Using dummy dataset for testing")
            dataset = create_dummy_dataset(
                vocab_size=config.vocab_size,
                seq_len=config.max_seq_len,
                batch_size=8,
                num_samples=1000
            )
        
        # Setup callbacks
        callbacks = setup_callbacks(checkpoint_dir="./checkpoints")
        
        # Training parameters
        epochs = 10
        steps_per_epoch = 100
        
        logger.info(f"Starting training for {epochs} epochs with {steps_per_epoch} steps per epoch")
        
        # Custom training loop with proper loss calculation
        @tf.function
        def train_step(batch):
            input_ids = batch['input_ids']
            # For language modeling, shift labels by one position
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]
            
            with tf.GradientTape() as tape:
                logits = model(inputs, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits, from_logits=True
                )
                loss = tf.reduce_mean(loss)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            return loss
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            num_steps = 0
            
            for step, batch in enumerate(dataset.take(steps_per_epoch)):
                loss = train_step(batch)
                epoch_loss += loss
                num_steps += 1
                
                if step % 20 == 0:
                    logger.info(f"Step {step}, Loss: {loss:.4f}")
            
            avg_loss = epoch_loss / num_steps
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
            # Manually trigger callbacks
            logs = {'loss': float(avg_loss)}
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(epoch, logs)
        
        # Save final model
        final_model_path = "./checkpoints/minigpt_final.keras"
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Test generation
        if model.tokenizer:
            logger.info("Testing text generation...")
            try:
                sample_text = model.generate_text(
                    "The future of artificial intelligence is",
                    max_length=50,
                    temperature=0.8
                )
                logger.info(f"Generated text: {sample_text}")
            except Exception as e:
                logger.warning(f"Text generation test failed: {e}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def load_and_test_model(model_path="./checkpoints/minigpt_final.keras"):
    """Load and test a saved model"""
    try:
        from enhanced_minigpt import EnhancedMiniGPT, ModelConfig
        
        # Create model with same config
        config = ModelConfig()
        model = EnhancedMiniGPT(config)
        model.build_model()
        
        # Load weights
        if model_path.endswith('.weights.h5'):
            model.load_weights(model_path)
            logger.info(f"Loaded model weights from {model_path}")
        else:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded full model from {model_path}")
        
        # Test the model
        test_input = tf.random.uniform((1, 64), minval=0, maxval=config.vocab_size, dtype=tf.int32)
        output = model(test_input)
        logger.info(f"Model test successful. Output shape: {output.shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

if __name__ == "__main__":
    # Set memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"GPU setup failed: {e}")
    
    # Run training
    train_model()