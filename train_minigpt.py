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

def create_dummy_dataset(vocab_size: int = 50257, seq_len: int = 128, batch_size: int = 2, num_samples: int = 1000):
    """Create a dummy dataset for testing with proper batching and infinite repeat."""
    def data_generator():
        while True:  # Make the generator infinite
            # Generate random input sequence
            input_ids = np.random.randint(0, vocab_size, size=(seq_len,), dtype=np.int32)
            yield input_ids
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_text_dataset(texts, tokenizer, seq_len=512, batch_size=8):
    """Create an infinite dataset from texts."""
    if not texts or not tokenizer:
        logger.warning("No texts or tokenizer provided")
        return None
        
    def data_generator():
        while True:  # Make the generator infinite
            for text in texts:
                if not isinstance(text, str) or len(text.strip()) == 0:
                    continue
                    
                try:
                    tokens = tokenizer.encode(text.strip())
                    stride = seq_len // 2
                    for i in range(0, len(tokens) - seq_len + 1, stride):
                        sequence = tokens[i:i + seq_len]
                        if len(sequence) == seq_len:
                            yield np.array(sequence, dtype=np.int32)
                except Exception as e:
                    logger.warning(f"Error processing text: {e}")
                    continue
    
    try:
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
        )
        
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return None

def load_conversation_datasets():
    """Load conversation datasets with proper cache clearing and error handling."""
    # Clear cache directory and disable caching completely
    try:
        import shutil
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info("Cleared HuggingFace cache directory")
    except Exception as e:
        logger.warning(f"Could not clear cache directory: {e}")
    
    # Disable all caching mechanisms
    disable_caching()
    datasets.config.HF_DATASETS_OFFLINE = False
    datasets.config.USE_AUTH_TOKEN = False
    
    # Set environment variables to prevent caching issues
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache'
    
    dataset_configs = [
        ("blended_skill_talk", "dialog"),
        ("daily_dialog", "dialogue")
    ]
    
    all_texts = []
    max_texts = 10000
    
    for dataset_name, field_name in dataset_configs:
        try:
            logger.info(f"Attempting to load {dataset_name}...")
            
            # Try multiple loading strategies
            dataset = None
            
            # Strategy 1: Force redownload with no verification
            try:
                dataset = datasets.load_dataset(
                    dataset_name, 
                    split="train[:1000]",
                    verification_mode="no_checks",
                    download_mode="force_redownload",
                    cache_dir=None,
                    keep_in_memory=True
                )
                logger.info(f"Successfully loaded {dataset_name} with force redownload")
            except Exception as e1:
                logger.warning(f"Strategy 1 failed for {dataset_name}: {e1}")
                
                # Strategy 2: Try without specifying split size
                try:
                    full_dataset = datasets.load_dataset(
                        dataset_name,
                        verification_mode="no_checks",
                        download_mode="force_redownload",
                        cache_dir=None,
                        keep_in_memory=True
                    )
                    # Take first 1000 samples manually
                    if hasattr(full_dataset, 'train') and full_dataset.train:
                        dataset = full_dataset.train.select(range(min(1000, len(full_dataset.train))))
                    elif hasattr(full_dataset, 'training') and full_dataset.training:
                        dataset = full_dataset.training.select(range(min(1000, len(full_dataset.training))))
                    else:
                        # Try to get any available split
                        available_splits = list(full_dataset.keys()) if hasattr(full_dataset, 'keys') else []
                        if available_splits:
                            first_split = full_dataset[available_splits[0]]
                            dataset = first_split.select(range(min(1000, len(first_split))))
                    
                    if dataset:
                        logger.info(f"Successfully loaded {dataset_name} with manual split selection")
                except Exception as e2:
                    logger.warning(f"Strategy 2 failed for {dataset_name}: {e2}")
                    
                    # Strategy 3: Try streaming mode
                    try:
                        streaming_dataset = datasets.load_dataset(
                            dataset_name,
                            streaming=True,
                            verification_mode="no_checks"
                        )
                        
                        # Convert streaming to regular dataset with limited samples
                        samples = []
                        if hasattr(streaming_dataset, 'train'):
                            train_stream = streaming_dataset.train
                        elif hasattr(streaming_dataset, 'training'):
                            train_stream = streaming_dataset.training
                        else:
                            available_splits = list(streaming_dataset.keys())
                            train_stream = streaming_dataset[available_splits[0]] if available_splits else None
                        
                        if train_stream:
                            for i, example in enumerate(train_stream):
                                if i >= 1000:
                                    break
                                samples.append(example)
                            
                            if samples:
                                dataset = datasets.Dataset.from_list(samples)
                                logger.info(f"Successfully loaded {dataset_name} with streaming mode")
                    except Exception as e3:
                        logger.warning(f"Strategy 3 failed for {dataset_name}: {e3}")
                        continue
            
            # Process the dataset if we successfully loaded it
            if dataset is not None:
                try:
                    # Extract text based on dataset structure
                    for example in dataset:
                        text_content = None
                        
                        # Try different field access patterns
                        if field_name in example and isinstance(example[field_name], list):
                            text_content = " ".join([str(turn) for turn in example[field_name]])
                        elif field_name in example and isinstance(example[field_name], str):
                            text_content = example[field_name]
                        elif 'text' in example:
                            text_content = example['text']
                        elif 'conversation' in example:
                            if isinstance(example['conversation'], list):
                                text_content = " ".join([str(turn) for turn in example['conversation']])
                            else:
                                text_content = str(example['conversation'])
                        
                        if text_content and len(text_content.strip()) > 10:
                            all_texts.append(text_content.strip())
                            if len(all_texts) >= max_texts:
                                break
                    
                    logger.info(f"Successfully extracted {len(all_texts)} texts from {dataset_name}")
                    
                except Exception as e:
                    logger.warning(f"Error processing {dataset_name}: {e}")
                    continue
            else:
                logger.warning(f"Could not load {dataset_name} with any strategy")
                
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            continue
    
    # Add comprehensive fallback data if no texts were loaded
    if not all_texts:
        logger.warning("No texts loaded from datasets, using comprehensive fallback data")
        fallback_texts = [
            "Hello, how are you today? I'm doing well, thank you for asking.",
            "What's your favorite color? I like blue because it reminds me of the sky.",
            "Do you enjoy reading books? Yes, I love reading science fiction and fantasy.",
            "What do you think about artificial intelligence? It's a fascinating field with lots of potential.",
            "How do you spend your free time? I enjoy hiking, reading, and learning new things.",
            "Tell me about your day. It's been productive and interesting, thank you for asking.",
            "What's the weather like today? It's sunny and warm, perfect for outdoor activities.",
            "Do you have any hobbies? I enjoy photography, cooking, and playing musical instruments.",
            "What's your favorite type of music? I appreciate various genres, from classical to contemporary.",
            "Can you recommend a good restaurant? There's a lovely Italian place downtown that I really enjoy.",
            "What are your plans for the weekend? I'm thinking of visiting a museum and then having dinner with friends.",
            "Do you like to travel? Yes, I find exploring new places and cultures very enriching.",
            "What's your favorite season? I love autumn because of the beautiful colors and crisp air.",
            "Do you prefer movies or books? Both have their merits, but I lean slightly towards books.",
            "What's the most interesting thing you've learned recently? I've been fascinated by advances in renewable energy.",
            "How do you stay motivated? I find setting small, achievable goals helps maintain momentum.",
            "What's your opinion on social media? It's a powerful tool for connection but requires mindful usage.",
            "Do you have any pets? I have a cat named Whiskers who's quite playful and affectionate.",
            "What's your favorite type of cuisine? I enjoy Mediterranean food for its fresh ingredients and flavors.",
            "How do you handle stress? I find meditation and regular exercise very helpful for managing stress."
        ]
        all_texts = fallback_texts * 50  # Repeat to have more variety
    
    logger.info(f"Total texts available for training: {len(all_texts)}")
    return all_texts[:max_texts]

def train_model(model, config):
    """Improved training function with better error handling."""
    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=1.0
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Create loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    
    # Training step with better error handling
    @tf.function
    def train_step(input_batch):
        # Create input and target sequences
        input_ids = input_batch[:, :-1]  # All tokens except the last
        targets = input_batch[:, 1:]     # All tokens except the first
        
        with tf.GradientTape() as tape:
            # Forward pass
            logits = model(input_ids, training=True)
            
            # Reshape for loss computation
            batch_size = tf.shape(logits)[0]
            seq_len = tf.shape(logits)[1]
            vocab_size = tf.shape(logits)[2]
            
            logits_flat = tf.reshape(logits, [-1, vocab_size])
            targets_flat = tf.reshape(targets, [-1])
            
            # Compute loss
            loss_per_token = loss_fn(targets_flat, logits_flat)
            loss = tf.reduce_mean(loss_per_token)
            
            # Scale loss for mixed precision
            scaled_loss = optimizer.get_scaled_loss(loss)
        
        # Compute gradients
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        
        # Check for NaN gradients
        finite_gradients = []
        for grad in gradients:
            if grad is not None:
                finite_gradients.append(tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad)))
            else:
                finite_gradients.append(None)
        
        # Apply gradients
        optimizer.apply_gradients(zip(finite_gradients, model.trainable_variables))
        
        # Calculate accuracy
        predictions = tf.argmax(logits_flat, axis=-1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(targets_flat, tf.cast(predictions, targets_flat.dtype)), tf.float32))
        
        return loss, accuracy
    
    # Create datasets
    logger.info("Creating datasets...")
    
    # Try to load conversation data
    texts = load_conversation_datasets()
    
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        logger.info("Using tokenizer to create dataset from texts")
        train_dataset = create_text_dataset(texts, model.tokenizer, config.seq_len, config.batch_size)
        val_dataset = create_text_dataset(texts[-100:], model.tokenizer, config.seq_len, config.batch_size)
    else:
        logger.info("No tokenizer available, using dummy dataset")
        train_dataset = None
        val_dataset = None
    
    # Fallback to dummy dataset
    if train_dataset is None:
        logger.info("Creating dummy datasets")
        train_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size, 1000)
        val_dataset = create_dummy_dataset(config.vocab_size, config.seq_len, config.batch_size, 100)
    
    # Create checkpoint directory
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training parameters
    num_epochs = 5  # Reduced for testing
    steps_per_epoch = 100  # Fixed number of steps per epoch
    validation_steps = 20
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("=" * 50)
        
        # Training phase
        train_losses = []
        train_accuracies = []
        
        try:
            train_iter = iter(train_dataset)
            
            for step in range(steps_per_epoch):
                try:
                    batch = next(train_iter)
                    if batch.shape[0] < config.batch_size:
                        logger.debug(f"Skipping incomplete batch of size {batch.shape[0]}")
                        continue
                        
                    loss, accuracy = train_step(batch)
                    
                    train_losses.append(float(loss.numpy()))
                    train_accuracies.append(float(accuracy.numpy()))
                    
                    if step % 20 == 0:
                        logger.info(f"Step {step}/{steps_per_epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                        
                except (StopIteration, tf.errors.OutOfRangeError):
                    logger.warning("Dataset exhausted, creating new iterator")
                    train_iter = iter(train_dataset)
                    continue  # Skip to next iteration instead of forcing a batch
                except Exception as e:
                    logger.error(f"Error in training step {step}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in training phase: {str(e)}")
            continue
            
        # Validation phase
        val_losses = []
        val_accuracies = []
        
        try:
            val_iter = iter(val_dataset)
            
            for step in range(validation_steps):
                try:
                    batch = next(val_iter)
                    
                    # Validation forward pass
                    input_ids = batch[:, :-1]
                    targets = batch[:, 1:]
                    
                    logits = model(input_ids, training=False)
                    
                    # Compute validation loss and accuracy
                    batch_size = tf.shape(logits)[0]
                    seq_len = tf.shape(logits)[1]
                    vocab_size = tf.shape(logits)[2]
                    
                    logits_flat = tf.reshape(logits, [-1, vocab_size])
                    targets_flat = tf.reshape(targets, [-1])
                    
                    val_loss = tf.reduce_mean(loss_fn(targets_flat, logits_flat))
                    predictions = tf.argmax(logits_flat, axis=-1)
                    val_accuracy = tf.reduce_mean(tf.cast(tf.equal(targets_flat, tf.cast(predictions, targets_flat.dtype)), tf.float32))
                    
                    val_losses.append(float(val_loss.numpy()))
                    val_accuracies.append(float(val_accuracy.numpy()))
                    
                except (StopIteration, tf.errors.OutOfRangeError):
                    val_iter = iter(val_dataset)
                    continue
                except Exception as e:
                    logger.error(f"Error in validation step {step}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in validation phase: {e}")
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        avg_train_acc = np.mean(train_accuracies) if train_accuracies else 0.0
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_val_acc = np.mean(val_accuracies) if val_accuracies else 0.0
        
        # Log epoch results
        logger.info("\nEpoch Summary:")
        logger.info("-" * 30)
        logger.info(f"Training   - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}")
        logger.info(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}")
        logger.info(f"Perplexity - Train: {np.exp(min(avg_train_loss, 10)):.2f}, Val: {np.exp(min(avg_val_loss, 10)):.2f}")
        logger.info("-" * 30)
        
        # Save checkpoint
        try:
            checkpoint_path = os.path.join(checkpoint_dir, f"minigpt_epoch_{epoch+1:02d}.weights.h5")
            model.save_weights(checkpoint_path)
            logger.info(f"Model saved to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    # Save final model
    try:
        final_path = os.path.join(checkpoint_dir, "minigpt_final.weights.h5")
        model.save_weights(final_path)
        logger.info(f"Final model saved to {final_path}")
    except Exception as e:
        logger.error(f"Error saving final model: {e}")
    
    return model

def simple_generate_text(model, prompt_tokens, max_length=50, temperature=0.8):
    """Simple text generation function."""
    if not hasattr(model, 'tokenizer') or model.tokenizer is None:
        return "Text generation requires a tokenizer."
    
    try:
        # Convert prompt to tokens if it's a string
        if isinstance(prompt_tokens, str):
            prompt_tokens = model.tokenizer.encode(prompt_tokens)
        
        # Ensure we have a reasonable prompt length
        if len(prompt_tokens) > 100:
            prompt_tokens = prompt_tokens[-100:]
        
        input_ids = tf.constant([prompt_tokens], dtype=tf.int32)
        
        for _ in range(max_length):
            # Get model prediction
            logits = model(input_ids, training=False)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample next token
            next_token = tf.random.categorical([next_token_logits], 1)[0, 0]
            
            # Add to sequence
            input_ids = tf.concat([input_ids, [[next_token]]], axis=1)
            
            # Stop at end token or max length
            if next_token == model.tokenizer.eos_token_id:
                break
        
        # Decode the generated sequence
        generated_tokens = input_ids[0].numpy().tolist()
        generated_text = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return f"Error: {e}"

def chat_loop(model):
    """Interactive chat loop with the model."""
    logger.info("Starting chat loop. Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                logger.info("Exiting chat loop.")
                break
            
            if not user_input:
                continue
            
            # Generate response
            response = simple_generate_text(model, user_input, max_length=30)
            print(f"\nAI: {response}")
            
        except KeyboardInterrupt:
            logger.info("\nExiting chat loop.")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            print("\nAI: Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    # Set memory growth for GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")

    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # Create model configuration with smaller parameters for testing
    config = ModelConfig(
        vocab_size=50257,
        max_seq_len=512,     # Reduced from 1024
        embed_dim=384,       # Reduced from 768
        num_heads=6,         # Reduced from 12
        num_layers=6,        # Reduced from 12
        ffn_dim=1536,        # Reduced from 3072
        dropout=0.1,
        use_custom_attention=True,
        use_rotary_embeddings=True,
        learning_rate=5e-4,  # Slightly higher learning rate
        batch_size=4,        # Reduced batch size
        seq_len=512          # Reduced sequence length
    )

    logger.info("Creating model...")
    model = EnhancedMiniGPT(config)

    # Build model
    dummy_input = tf.random.uniform((2, config.seq_len), maxval=config.vocab_size, dtype=tf.int32)
    _ = model(dummy_input)
    
    logger.info(f"Model created with {model.count_params():,} parameters")

    # Train model
    logger.info("Starting training...")
    try:
        model = train_model(model, config)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")

    # Start chat loop
    logger.info("Starting chat interface...")
    chat_loop(model)