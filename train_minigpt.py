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

def create_dummy_dataset(vocab_size=50257, seq_len=512, batch_size=1, num_samples=1000):  # Changed from 8 to 1
    """Create a dummy dataset for testing"""
    def data_generator():
        for _ in range(num_samples):
            # Generate random input sequence
            input_seq = np.random.randint(0, vocab_size, size=seq_len, dtype=np.int32)
            yield {'input_ids': input_seq}
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature={
            'input_ids': tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
        }
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_text_dataset(text_file_path, tokenizer, seq_len=512, batch_size=1):  # Changed from 8 to 1
    """Create a dataset from a text file."""
    def data_generator():
        with open(text_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Tokenize text
                tokens = tokenizer.encode(line.strip())
                if len(tokens) < seq_len:
                    continue
                
                # Create sequences with overlap
                for i in range(0, len(tokens) - seq_len, seq_len // 2):
                    yield {'input_ids': tokens[i:i + seq_len]}
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature={
            'input_ids': tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
        }
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

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
        
        # Format filepath - Fix the formatting issue
        try:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
        except (KeyError, ValueError) as e:
            # Fallback to simple epoch-based naming
            filepath = f"{self.filepath}_epoch_{epoch + 1:02d}"
        
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
            filepath=os.path.join(checkpoint_dir, "minigpt_epoch_{epoch:02d}"),
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
        ("conv_ai_2", None),  # Use None for default config
        ("blended_skill_talk", None),
        ("daily_dialog", None)
    ]
    
    all_texts = []
    
    for dataset_name, config_name in dataset_configs:
        try:
            # Load dataset
            if config_name:
                dataset = datasets.load_dataset(dataset_name, config_name, split="train")
            else:
                dataset = datasets.load_dataset(dataset_name, split="train")
            logger.info(f"Loaded {dataset_name} dataset")
            
            # Extract conversations
            texts = []
            for example in dataset:
                if isinstance(example, dict):
                    # Handle different dataset formats
                    if "dialog" in example:
                        dialog = example["dialog"]
                        if isinstance(dialog, list):
                            texts.extend([str(turn) for turn in dialog])
                    elif "conversation" in example:
                        conv = example["conversation"]
                        if isinstance(conv, list):
                            texts.extend([str(turn) for turn in conv])
                    elif "text" in example:
                        texts.append(str(example["text"]))
                    elif "free_messages" in example:
                        messages = example["free_messages"]
                        if isinstance(messages, list):
                            texts.extend([str(msg) for msg in messages])
                    elif "utterances" in example:
                        utterances = example["utterances"]
                        if isinstance(utterances, list):
                            texts.extend([str(utt) for utt in utterances])
            
            all_texts.extend(texts)
            logger.info(f"Added {len(texts)} texts from {dataset_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
    
    return all_texts

def create_dataset_from_texts(texts, tokenizer, seq_len, batch_size = 1):
    """Create a TensorFlow dataset from a list of texts."""
    try:
        if not texts:
            logger.warning("No texts provided for dataset creation")
            return None
            
        # Filter out non-string texts and empty strings
        texts = [str(text).strip() for text in texts if text and str(text).strip()]
        if not texts:
            logger.warning("No valid texts after filtering")
            return None
            
        # Create sequences
        sequences = []
        for text in texts:
            try:
                # Tokenize text
                tokens = tokenizer.encode(text)
                if not tokens or len(tokens) < 2:  # Need at least 2 tokens
                    continue
                    
                # Create overlapping sequences of length seq_len
                for i in range(0, len(tokens) - seq_len + 1, seq_len // 2):
                    sequence = tokens[i:i + seq_len]
                    if len(sequence) == seq_len:  # Only add complete sequences
                        sequences.append(sequence)
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                continue
        
        if not sequences:
            logger.warning("No valid sequences created from texts")
            return None
            
        logger.info(f"Created {len(sequences)} sequences from {len(texts)} texts")
        
        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(sequences)
        dataset = dataset.map(lambda x: {'input_ids': x})
        dataset = dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return None

def train_model(model, config):
    """Train the model."""
    # Create optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=1.0  # Add gradient clipping
    )
    
    # Create metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_perplexity = tf.keras.metrics.Mean(name='train_perplexity')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_perplexity = tf.keras.metrics.Mean(name='val_perplexity')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    
    # Create loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    # Define training step - Fixed shape handling
    @tf.function
    def train_step(batch):
        input_ids = batch['input_ids']
        
        # Create input and target sequences
        inputs = input_ids[:, :-1]  # All tokens except the last one
        targets = input_ids[:, 1:]  # All tokens except the first one
        
        with tf.GradientTape() as tape:
            # Forward pass
            logits = model(inputs, training=True)
            
            # Compute loss - ensure shapes match
            batch_size = tf.shape(targets)[0]
            seq_len = tf.shape(targets)[1]
            
            # Flatten targets for loss computation
            targets_flat = tf.reshape(targets, [-1])
            logits_flat = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
            
            # Compute loss
            loss_per_token = loss_fn(targets_flat, logits_flat)
            loss = tf.reduce_mean(loss_per_token)
            
            # Add regularization losses
            if model.losses:
                loss += tf.reduce_sum(model.losses)
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        train_loss.update_state(loss)
        train_perplexity.update_state(tf.exp(tf.minimum(loss, 10.0)))  # Cap to prevent overflow
        train_accuracy.update_state(targets, logits)
        
        return loss
    
    # Define validation step - Fixed shape handling
    @tf.function
    def val_step(batch):
        input_ids = batch['input_ids']
        
        # Create input and target sequences
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        
        # Forward pass
        logits = model(inputs, training=False)
        
        # Compute loss
        targets_flat = tf.reshape(targets, [-1])
        logits_flat = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
        
        loss_per_token = loss_fn(targets_flat, logits_flat)
        loss = tf.reduce_mean(loss_per_token)
        
        # Update metrics
        val_loss.update_state(loss)
        val_perplexity.update_state(tf.exp(tf.minimum(loss, 10.0)))
        val_accuracy.update_state(targets, logits)
        
        return loss
    
    # Create checkpoint
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    # Create summary writers
    train_log_dir = './logs/train'
    val_log_dir = './logs/val'
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(val_log_dir, exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    # Load datasets
    logger.info("Loading conversation datasets...")
    texts = load_conversation_datasets()
    
    # Check if we have a tokenizer
    if not hasattr(model, 'tokenizer') or model.tokenizer is None:
        logger.warning("No tokenizer available, using dummy dataset")
        train_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size)
        val_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size, num_samples=100)
    else:
        # Create datasets
        if texts:
            dataset = create_dataset_from_texts(texts, model.tokenizer, config.seq_len, config.batch_size)
            if dataset is None:
                logger.warning("Failed to create dataset from texts, using dummy dataset")
                train_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size)
                val_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size, num_samples=100)
            else:
                # Split into train and validation
                dataset_list = list(dataset)
                total_size = len(dataset_list)
                train_size = int(0.9 * total_size)
                
                train_dataset = tf.data.Dataset.from_tensor_slices(dataset_list[:train_size])
                val_dataset = tf.data.Dataset.from_tensor_slices(dataset_list[train_size:])
                
                # Rebatch the datasets
                train_dataset = train_dataset.unbatch().batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
                val_dataset = val_dataset.unbatch().batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            logger.warning("No texts loaded, using dummy dataset")
            train_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size)
            val_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size, num_samples=100)
    
    # Training loop
    num_epochs = 10
    step_count = 0
    
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
            batch_count = 0
            for batch in train_dataset:
                if batch is None:
                    continue
                    
                input_ids = batch['input_ids']
                if input_ids is None or tf.size(input_ids) == 0:
                    continue
                
                # Ensure input has enough tokens
                if tf.shape(input_ids)[1] < 2:
                    continue
                
                # Train step
                loss = train_step(batch)
                step_count += 1
                batch_count += 1
                
                # Log progress every 100 batches
                if batch_count % 100 == 0:
                    logger.info(f"Epoch {epoch + 1}, Batch {batch_count}, Loss: {float(loss):.4f}")
                    
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
                
                # Ensure input has enough tokens
                if tf.shape(input_ids)[1] < 2:
                    continue
                
                val_step(val_batch)
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
                f"Train Loss: {float(train_loss.result()):.4f}, "
                f"Train Perplexity: {float(train_perplexity.result()):.4f}, "
                f"Train Accuracy: {float(train_accuracy.result()):.4f}, "
                f"Val Loss: {float(val_loss.result()):.4f}, "
                f"Val Perplexity: {float(val_perplexity.result()):.4f}, "
                f"Val Accuracy: {float(val_accuracy.result()):.4f}"
            )
        except Exception as e:
            logger.error(f"Error printing metrics: {e}")
        
        # Save checkpoint every epoch
        try:
            checkpoint.save(file_prefix=checkpoint_prefix)
            logger.info(f"Checkpoint saved for epoch {epoch + 1}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    # Save final model
    try:
        final_weights_path = './checkpoints/minigpt_final.weights.h5'
        model.save_weights(final_weights_path)
        logger.info(f"Training completed. Model saved to {final_weights_path}")
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
            if hasattr(model, 'chat'):
                response = model.chat(user_input)
            else:
                response = "Chat functionality not implemented in model."
            print(f"\nAI: {response}")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            print("\nAI: I apologize, but I encountered an error. Please try again.")

if __name__ == "__main__":
    # Create model configuration
    config = ModelConfig(
        vocab_size=50257,
        max_seq_len=512,
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        ffn_dim=1536,
        dropout=0.1,
        use_custom_attention=True,
        use_rotary_embeddings=True,
        learning_rate=5e-4,
        batch_size=1,  # Changed from 8 to 1
        seq_len=512
    )
    
    # Create model
    logger.info("Creating model...")
    model = EnhancedMiniGPT(config)
    
    # Build model by calling it with dummy input
    dummy_input = tf.random.uniform((1, config.seq_len), maxval=config.vocab_size, dtype=tf.int32)
    _ = model(dummy_input)
    logger.info(f"Model created successfully with {model.count_params():,} parameters")
    
    # Train model
    logger.info("Starting training...")
    model = train_model(model, config)
    
    # Start chat loop
    chat_loop(model)