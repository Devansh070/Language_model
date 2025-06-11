import tensorflow as tf
import numpy as np
import os
import logging
from datetime import datetime
import json
from pathlib import Path
import datasets
from datasets import disable_caching
from minigpt_transformer import (
    EnhancedMiniGPT,
    ModelConfig,
    CustomMultiHeadAttention,
    MultiHeadAttention,
    RotaryPositionalEmbedding,
    FeedForward,
    TransformerBlock
)

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
                self.model.save_weights(filepath, save_format='h5')
            else:
                # Save full model
                if not filepath.endswith('.keras'):
                    filepath = filepath + '.keras'
                self.model.save(filepath, save_format='keras')
            
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

def load_conversation_datasets():
    """Load conversation datasets from HuggingFace."""
    disable_caching()
    
    # Define datasets to load
    dataset_configs = [
        ("conversation", "conversation"),
        ("daily_dialog", "daily_dialog"),
        ("conv_ai_2", "conv_ai_2"),
        ("blended_skill_talk", "blended_skill_talk")
    ]
    
    all_texts = []
    
    for dataset_name, config_name in dataset_configs:
        try:
            # Load dataset
            dataset = datasets.load_dataset(dataset_name, config_name, split="train")
            logger.info(f"Loaded {dataset_name} dataset")
            
            # Extract conversations
            texts = []
            for example in dataset:
                if isinstance(example, dict):
                    # Handle different dataset formats
                    if "dialog" in example:
                        dialog = example["dialog"]
                        if isinstance(dialog, list):
                            texts.extend(dialog)
                    elif "conversation" in example:
                        conv = example["conversation"]
                        if isinstance(conv, list):
                            texts.extend(conv)
                    elif "text" in example:
                        texts.append(example["text"])
                    elif "free_messages" in example:
                        messages = example["free_messages"]
                        if isinstance(messages, list):
                            texts.extend(messages)
            
            all_texts.extend(texts)
            logger.info(f"Added {len(texts)} texts from {dataset_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
    
    return all_texts

def create_dataset_from_texts(texts, tokenizer, seq_len, batch_size):
    """Create a TensorFlow dataset from a list of texts."""
    try:
        if not texts:
            logger.warning("No texts provided for dataset creation")
            return None
            
        # Filter out non-string texts
        texts = [text for text in texts if isinstance(text, str)]
        if not texts:
            logger.warning("No valid texts after filtering")
            return None
            
        # Create sequences
        sequences = []
        for text in texts:
            try:
                # Tokenize text
                tokens = tokenizer.encode(text)
                if not tokens:
                    continue
                    
                # Create sequences of length seq_len
                for i in range(0, len(tokens) - seq_len + 1, seq_len):
                    sequence = tokens[i:i + seq_len]
                    if len(sequence) == seq_len:  # Only add complete sequences
                        sequences.append(sequence)
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                continue
        
        if not sequences:
            logger.warning("No valid sequences created from texts")
            return None
            
        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(sequences)
        dataset = dataset.map(lambda x: {'input_ids': x})
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return None

def train_model(model, config):
    """Train the model."""
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    # Create metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_perplexity = tf.keras.metrics.Mean(name='train_perplexity')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_perplexity = tf.keras.metrics.Mean(name='val_perplexity')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    
    # Create loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Define training step
    @tf.function
    def train_step(input_ids, labels):
        with tf.GradientTape() as tape:
            # Forward pass
            logits = model(input_ids, training=True)
            
            # Compute loss
            loss = loss_fn(labels, logits)
            
            # Add regularization losses
            loss += tf.reduce_sum(model.losses)
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        train_loss.update_state(loss)
        train_perplexity.update_state(tf.exp(loss))
        train_accuracy.update_state(labels, logits)
        
        return loss
    
    # Define validation step
    @tf.function
    def val_step(input_ids, labels):
        # Forward pass
        logits = model(input_ids, training=False)
        
        # Compute loss
        loss = loss_fn(labels, logits)
        
        # Update metrics
        val_loss.update_state(loss)
        val_perplexity.update_state(tf.exp(loss))
        val_accuracy.update_state(labels, logits)
        
        return loss
    
    # Create checkpoint
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    # Create summary writers
    train_log_dir = './logs/train'
    val_log_dir = './logs/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    # Load datasets
    texts = load_conversation_datasets()
    
    # Check if we have a tokenizer
    if not hasattr(model, 'tokenizer') or model.tokenizer is None:
        logger.warning("No tokenizer available, using dummy dataset")
        train_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size)
        val_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size)
    else:
        # Create datasets
        dataset = create_dataset_from_texts(texts, model.tokenizer, config.seq_len, config.batch_size)
        if dataset is None:
            logger.warning("Failed to create dataset from texts, using dummy dataset")
            train_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size)
            val_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size)
        else:
            # Split into train and validation
            total_size = sum(1 for _ in dataset)
            train_size = int(0.9 * total_size)
            train_dataset = dataset.take(train_size)
            val_dataset = dataset.skip(train_size)
    
    # Training loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Reset metrics
        train_loss.reset_state()
        train_perplexity.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_perplexity.reset_state()
        val_accuracy.reset_state()
        
        # Training
        try:
            for batch in train_dataset:
                if batch is None:
                    continue
                    
                input_ids = batch['input_ids']
                if input_ids is None or tf.size(input_ids) == 0:
                    continue
                    
                labels = input_ids[:, 1:]  # Shift labels by one position
                
                # Train step
                loss = train_step(input_ids, labels)
                
                # Log progress
                if train_accuracy.count % 100 == 0:
                    logger.info(f"Processed {train_accuracy.count} batches, Loss: {loss:.4f}")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            continue
        
        # Validation
        try:
            for val_batch in val_dataset:
                if val_batch is None:
                    continue
                    
                input_ids = val_batch['input_ids']
                if input_ids is None or tf.size(input_ids) == 0:
                    continue
                    
                labels = input_ids[:, 1:]
                val_step(input_ids, labels)
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            continue
        
        # Log metrics
        try:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('perplexity', train_perplexity.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=epoch)
                tf.summary.scalar('perplexity', val_perplexity.result(), step=epoch)
                tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
        
        # Print metrics
        try:
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss.result():.4f}, "
                f"Train Perplexity: {train_perplexity.result():.4f}, "
                f"Train Accuracy: {train_accuracy.result():.4f}, "
                f"Val Loss: {val_loss.result():.4f}, "
                f"Val Perplexity: {val_perplexity.result():.4f}, "
                f"Val Accuracy: {val_accuracy.result():.4f}"
            )
        except Exception as e:
            logger.error(f"Error printing metrics: {e}")
        
        # Save checkpoint
        try:
            checkpoint.save(file_prefix=checkpoint_prefix)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    # Save final model
    try:
        model.save_weights('./checkpoints/minigpt_final.weights.h5')
        logger.info("Training completed. Model saved to ./checkpoints/minigpt_final.weights.h5")
    except Exception as e:
        logger.error(f"Error saving final model: {e}")
    
    return model

def chat_loop(model):
    """Interactive chat loop with the model."""
    logger.info("Starting chat loop. Type 'quit' to exit.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for quit command
        if user_input.lower() == 'quit':
            logger.info("Exiting chat loop.")
            break
        
        try:
            # Generate response
            response = model.chat(user_input)
            print(f"\nAI: {response}")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            print("\nAI: I apologize, but I encountered an error. Please try again.")

if __name__ == "__main__":
    # Create model configuration
    config = ModelConfig(
        vocab_size=50257,  # GPT-2 vocabulary size
        max_seq_len=1024,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        ffn_dim=3072,
        dropout=0.1,
        use_custom_attention=True,
        use_rotary_embeddings=True,
        learning_rate=1e-4,
        batch_size=32,
        seq_len=1024
    )
    
    # Create model
    model = EnhancedMiniGPT(config)
    
    # Train model
    model = train_model(model, config)
    
    # Start chat loop
    chat_loop(model)