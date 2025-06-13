import tensorflow as tf
import numpy as np
import os
import logging
from datetime import datetime
import json
from pathlib import Path
import datasets
from datasets import disable_caching
import time
from tqdm import tqdm
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

class TrainingMetrics:
    """Class to track and display training metrics."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.accuracies = []
        self.perplexities = []
        self.start_time = time.time()
    
    def update(self, loss, accuracy):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        perplexity = np.exp(min(loss, 10))  # Cap to prevent overflow
        self.perplexities.append(perplexity)
    
    def get_averages(self):
        if not self.losses:
            return 0.0, 0.0, 0.0
        return np.mean(self.losses), np.mean(self.accuracies), np.mean(self.perplexities)
    
    def get_current(self):
        if not self.losses:
            return 0.0, 0.0, 0.0
        return self.losses[-1], self.accuracies[-1], self.perplexities[-1]

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
    """Load conversation datasets with improved error handling and fallback strategies."""
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
    
    # Simplified dataset configs with better fallback
    dataset_configs = [
        ("blended_skill_talk", "dialog"),
        ("daily_dialog", "dialogue")
    ]
    
    all_texts = []
    max_texts = 5000  # Reduced for faster loading
    
    for dataset_name, field_name in dataset_configs:
        try:
            logger.info(f"Attempting to load {dataset_name}...")
            
            # Try the most reliable loading strategy first
            dataset = None
            
            try:
                # Strategy 1: Simple load with minimal configuration
                dataset = datasets.load_dataset(
                    dataset_name,
                    split="train",
                    trust_remote_code=True,
                    verification_mode="no_checks"
                )
                
                # Take only first 1000 samples to avoid memory issues
                if len(dataset) > 1000:
                    dataset = dataset.select(range(1000))
                    
                logger.info(f"Successfully loaded {dataset_name} with simple strategy")
                
            except Exception as e1:
                logger.warning(f"Simple strategy failed for {dataset_name}: {e1}")
                
                # Strategy 2: Try with different split configurations
                try:
                    # Try different split names
                    possible_splits = ['train', 'training', 'dialogue']
                    full_dataset = datasets.load_dataset(
                        dataset_name,
                        trust_remote_code=True,
                        verification_mode="no_checks"
                    )
                    
                    dataset = None
                    for split_name in possible_splits:
                        if hasattr(full_dataset, split_name):
                            split_data = getattr(full_dataset, split_name)
                            if split_data is not None:
                                dataset = split_data.select(range(min(1000, len(split_data))))
                                logger.info(f"Successfully loaded {dataset_name} using split '{split_name}'")
                                break
                    
                    if dataset is None:
                        # Try to get any available split
                        available_splits = list(full_dataset.keys()) if hasattr(full_dataset, 'keys') else []
                        if available_splits:
                            first_split = full_dataset[available_splits[0]]
                            dataset = first_split.select(range(min(1000, len(first_split))))
                            logger.info(f"Successfully loaded {dataset_name} using first available split")
                    
                except Exception as e2:
                    logger.warning(f"Split strategy failed for {dataset_name}: {e2}")
                    continue
            
            # Process the dataset if we successfully loaded it
            if dataset is not None:
                try:
                    logger.info(f"Processing {len(dataset)} examples from {dataset_name}")
                    
                    # Extract text based on dataset structure
                    for i, example in enumerate(dataset):
                        if i % 100 == 0:
                            logger.info(f"Processing example {i}/{len(dataset)} from {dataset_name}")
                            
                        text_content = None
                        
                        # Try different field access patterns
                        if field_name in example:
                            if isinstance(example[field_name], list):
                                # Handle list of dialogue turns
                                text_content = " ".join([str(turn) for turn in example[field_name] if turn])
                            elif isinstance(example[field_name], str):
                                text_content = example[field_name]
                        elif 'text' in example:
                            text_content = example['text']
                        elif 'conversation' in example:
                            if isinstance(example['conversation'], list):
                                text_content = " ".join([str(turn) for turn in example['conversation'] if turn])
                            else:
                                text_content = str(example['conversation'])
                        elif 'utterances' in example:
                            if isinstance(example['utterances'], list):
                                text_content = " ".join([str(utt) for utt in example['utterances'] if utt])
                        
                        # Clean and validate text
                        if text_content:
                            text_content = text_content.strip()
                            if len(text_content) > 20:  # Minimum length requirement
                                all_texts.append(text_content)
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
    
    # Add comprehensive fallback data if insufficient texts were loaded
    if len(all_texts) < 100:
        logger.warning(f"Only {len(all_texts)} texts loaded from datasets, adding fallback data")
        fallback_texts = [
            "Hello, how are you today? I'm doing well, thank you for asking. What about you?",
            "What's your favorite color? I like blue because it reminds me of the sky and ocean.",
            "Do you enjoy reading books? Yes, I love reading science fiction and fantasy novels.",
            "What do you think about artificial intelligence? It's a fascinating field with lots of potential for helping people.",
            "How do you spend your free time? I enjoy hiking, reading, cooking, and learning new programming languages.",
            "Tell me about your day. It's been productive and interesting, I've learned several new things today.",
            "What's the weather like today? It's sunny and warm, perfect for outdoor activities and spending time in nature.",
            "Do you have any hobbies? I enjoy photography, cooking, playing musical instruments, and gardening.",
            "What's your favorite type of music? I appreciate various genres, from classical to jazz to contemporary rock.",
            "Can you recommend a good restaurant? There's a lovely Italian place downtown that serves amazing pasta dishes.",
            "What are your plans for the weekend? I'm thinking of visiting a museum and then having dinner with friends.",
            "Do you like to travel? Yes, I find exploring new places and cultures very enriching and educational.",
            "What's your favorite season? I love autumn because of the beautiful colors and crisp, fresh air.",
            "Do you prefer movies or books? Both have their merits, but I lean slightly towards books for their depth.",
            "What's the most interesting thing you've learned recently? I've been fascinated by advances in renewable energy technology.",
            "How do you stay motivated? I find setting small, achievable goals helps maintain momentum and progress.",
            "What's your opinion on social media? It's a powerful tool for connection but requires mindful and balanced usage.",
            "Do you have any pets? I have a cat named Whiskers who's quite playful and brings joy to everyday life.",
            "What's your favorite type of cuisine? I enjoy Mediterranean food for its fresh ingredients and healthy flavors.",
            "How do you handle stress? I find meditation, regular exercise, and talking to friends very helpful for managing stress and anxiety.",
            "What's your dream job? I'd love to work in a field that combines creativity with technology to solve meaningful problems.",
            "What book are you currently reading? I'm reading a fascinating book about the history of human civilization and cultural development.",
            "What's your favorite way to exercise? I enjoy a mix of cardio activities like running and strength training at the gym.",
            "How do you learn new skills? I prefer a combination of online courses, practical projects, and learning from experienced mentors.",
            "What's your favorite holiday? I love the winter holidays because they bring families together and create warm, memorable experiences."
        ]
        # Duplicate fallback texts to have sufficient training data
        multiplier = max(1, (max_texts - len(all_texts)) // len(fallback_texts) + 1)
        all_texts.extend(fallback_texts * multiplier)
    
    # Ensure we don't exceed the maximum
    all_texts = all_texts[:max_texts]
    logger.info(f"Total texts available for training: {len(all_texts)}")
    return all_texts

def train_model(model, config):
    """Enhanced training function with comprehensive progress tracking and metrics display."""
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
    num_epochs = 5
    steps_per_epoch = 100
    validation_steps = 20
    
    # Initialize metrics tracking
    train_metrics = TrainingMetrics()
    val_metrics = TrainingMetrics()
    
    # Training history for plotting
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'train_perplexity': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_perplexity': []
    }
    
    logger.info("🚀 STARTING TRAINING")
    logger.info("=" * 80)
    logger.info(f"📊 Configuration:")
    logger.info(f"   • Epochs: {num_epochs}")
    logger.info(f"   • Steps per epoch: {steps_per_epoch}")
    logger.info(f"   • Batch size: {config.batch_size}")
    logger.info(f"   • Learning rate: {config.learning_rate}")
    logger.info(f"   • Model parameters: {model.count_params():,}")
    logger.info("=" * 80)
    
    # Global training start time
    global_start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🎯 EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Reset metrics for this epoch
        train_metrics.reset()
        
        # Training phase with progress bar
        train_iter = iter(train_dataset)
        
        print("📈 Training Phase:")
        train_pbar = tqdm(range(steps_per_epoch), desc="Training", ncols=100)
        
        for step in train_pbar:
            try:
                batch = next(train_iter)
                if batch.shape[0] < config.batch_size:
                    continue
                    
                loss, accuracy = train_step(batch)
                
                # Update metrics
                train_metrics.update(float(loss.numpy()), float(accuracy.numpy()))
                
                # Update progress bar with current metrics
                current_loss, current_acc, current_perp = train_metrics.get_current()
                avg_loss, avg_acc, avg_perp = train_metrics.get_averages()
                
                train_pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.3f}',
                    'Perp': f'{current_perp:.2f}',
                    'Avg_Loss': f'{avg_loss:.4f}'
                })
                
            except (StopIteration, tf.errors.OutOfRangeError):
                train_iter = iter(train_dataset)
                continue
            except Exception as e:
                logger.error(f"Error in training step {step}: {e}")
                continue
        
        train_pbar.close()
        
        # Validation phase with progress bar
        val_metrics.reset()
        val_iter = iter(val_dataset)
        
        print("📊 Validation Phase:")
        val_pbar = tqdm(range(validation_steps), desc="Validation", ncols=100)
        
        for step in val_pbar:
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
                
                # Update metrics
                val_metrics.update(float(val_loss.numpy()), float(val_accuracy.numpy()))
                
                # Update progress bar
                current_loss, current_acc, current_perp = val_metrics.get_current()
                avg_loss, avg_acc, avg_perp = val_metrics.get_averages()
                
                val_pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.3f}',
                    'Perp': f'{current_perp:.2f}',
                    'Avg_Loss': f'{avg_loss:.4f}'
                })
                
            except (StopIteration, tf.errors.OutOfRangeError):
                val_iter = iter(val_dataset)
                continue
            except Exception as e:
                logger.error(f"Error in validation step {step}: {e}")
                continue
        
        val_pbar.close()
        
        # Calculate epoch metrics
        train_loss, train_acc, train_perp = train_metrics.get_averages()
        val_loss, val_acc, val_perp = val_metrics.get_averages()
        
        # Store metrics in history
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(train_loss)
        training_history['train_accuracy'].append(train_acc)
        training_history['train_perplexity'].append(train_perp)
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_acc)
        training_history['val_perplexity'].append(val_perp)
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        total_duration = time.time() - global_start_time
        
        # Display comprehensive epoch summary
        print(f"\n🏆 EPOCH {epoch + 1} RESULTS:")
        print(f"{'─'*50}")
        print(f"⏱️  Duration: {epoch_duration:.1f}s | Total: {total_duration:.1f}s")
        print(f"🔥 Training   → Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | Perp: {train_perp:.2f}")
        print(f"✅ Validation → Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | Perp: {val_perp:.2f}")
        
        # Show improvement indicators
        if epoch > 0:
            prev_train_loss = training_history['train_loss'][-2]
            prev_val_loss = training_history['val_loss'][-2]
            train_improvement = prev_train_loss - train_loss
            val_improvement = prev_val_loss - val_loss
            
            train_indicator = "📈" if train_improvement > 0 else "📉"
            val_indicator = "📈" if val_improvement > 0 else "📉"
            
            print(f"📊 Changes    → Train: {train_indicator} {train_improvement:+.4f} | Val: {val_indicator} {val_improvement:+.4f}")
        
        print(f"{'─'*50}")
        
        # Save checkpoint
        try:
            checkpoint_path = os.path.join(checkpoint_dir, f"minigpt_epoch_{epoch+1:02d}.weights.h5")
            model.save_weights(checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Error saving checkpoint: {e}")
    
    # Training completion summary
    total_time = time.time() - global_start_time
    print(f"\n{'🎉'*20}")
    print(f"✨ TRAINING COMPLETED! ✨")
    print(f"{'🎉'*20}")
    print(f"⏰ Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"🏁 Final metrics:")
    print(f"   • Training Loss: {training_history['train_loss'][-1]:.4f}")
    print(f"   • Training Accuracy: {training_history['train_accuracy'][-1]:.3f}")
    print(f"   • Training Perplexity: {training_history['train_perplexity'][-1]:.2f}")
    print(f"   • Validation Loss: {training_history['val_loss'][-1]:.4f}")
    print(f"   • Validation Accuracy: {training_history['val_accuracy'][-1]:.3f}")
    print(f"   • Validation Perplexity: {training_history['val_perplexity'][-1]:.2f}")
    
    # Save final model
    try:
        final_path = os.path.join(checkpoint_dir, "minigpt_final.weights.h5")
        model.save_weights(final_path)
        print(f"💾 Final model saved: {final_path}")
    except Exception as e:
        logger.error(f"❌ Error saving final model: {e}")
    
    # Save training history
    try:
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"📊 Training history saved: {history_path}")
    except Exception as e:
        logger.error(f"❌ Error saving training history: {e}")
    
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
    print("\n🤖 Starting chat interface!")
    print("💬 Type your message and press Enter to chat.")
    print("🚪 Type 'quit', 'exit', or 'bye' to exit.")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for chatting!")
                break
            
            if not user_input:
                continue
            
            print("🤔 AI is thinking...")
            # Generate response
            response = simple_generate_text(model, user_input, max_length=30)
            print(f"🤖 AI: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            print("🤖 AI: Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    # Set memory growth for GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🎮 GPU setup complete: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        print("💻 No GPU detected, using CPU")

    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("⚡ Mixed precision enabled")

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

    print("\n🏗️  INITIALIZING MODEL")
    print("=" * 50)
    logger.info("Creating model...")
    model = EnhancedMiniGPT(config)

    # Build model
    print("🔧 Building model architecture...")
    dummy_input = tf.random.uniform((2, config.seq_len), maxval=config.vocab_size, dtype=tf.int32)
    _ = model(dummy_input)
    
    print(f"✅ Model created successfully!")
    print(f"📊 Model parameters: {model.count_params():,}")
    print(f"🧠 Architecture: {config.num_layers} layers, {config.num_heads} heads, {config.embed_dim} dimensions")

    # Train model
    print("\n🎯 INITIATING TRAINING SEQUENCE")
    print("=" * 50)
    try:
        model = train_model(model, config)
        print("🎊 Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"💥 Training encountered an error: {e}")
        print("🔧 Check the logs above for more details.")

    # Start chat loop
    print("\n🚀 LAUNCHING CHAT INTERFACE")
    print("=" * 50)
    try:
        chat_loop(model)
    except Exception as e:
        logger.error(f"❌ Chat interface error: {e}")
        print(f"💥 Chat interface encountered an error: {e}")
    
    print("\n🎯 PROGRAM COMPLETED")
    print("=" * 30)
    print("Thank you for using MiniGPT! 🤖✨")