import tensorflow as tf
import numpy as np
import os
import logging
from datetime import datetime
import json
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTokenizer:
    """Simple byte-pair encoding style tokenizer"""
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        # Create a simple vocabulary
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Add special tokens
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        for i, token in enumerate(special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token
        
        # Add ASCII characters
        for i in range(32, 127):  # Printable ASCII
            char = chr(i)
            token_id = len(self.char_to_id)
            if token_id < vocab_size:
                self.char_to_id[char] = token_id
                self.id_to_char[token_id] = char
        
        # Fill remaining vocabulary with dummy tokens
        for i in range(len(self.char_to_id), vocab_size):
            self.id_to_char[i] = f'<token_{i}>'
    
    def encode(self, text):
        """Encode text to token IDs"""
        if not text:
            return []
        
        tokens = []
        tokens.append(self.char_to_id.get('<bos>', 2))  # Start token
        
        for char in text:
            token_id = self.char_to_id.get(char, self.char_to_id.get('<unk>', 1))
            tokens.append(token_id)
        
        tokens.append(self.char_to_id.get('<eos>', 3))  # End token
        return tokens
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        text = ""
        for token_id in token_ids:
            if token_id == self.char_to_id.get('<bos>', 2):
                continue
            elif token_id == self.char_to_id.get('<eos>', 3):
                break
            elif token_id == self.char_to_id.get('<pad>', 0):
                continue
            else:
                char = self.id_to_char.get(token_id, '<unk>')
                if not char.startswith('<token_'):
                    text += char
        return text

def create_simple_text_dataset(vocab_size=50257, seq_len=512, batch_size=1, num_samples=1000):
    """Create a simple text dataset with meaningful patterns"""
    tokenizer = SimpleTokenizer(vocab_size)
    
    # Create more diverse training texts for longer sequences
    training_texts = [
        "Hello world! How are you today? I hope you are doing well and having a great time learning about natural language processing.",
        "The quick brown fox jumps over the lazy dog. This sentence contains all the letters of the alphabet and is commonly used for testing.",
        "Machine learning is fascinating and powerful. It enables computers to learn patterns from data without being explicitly programmed for every task.",
        "Natural language processing enables computers to understand text. It combines computational linguistics with machine learning to process human language.",
        "Deep learning models can generate human-like text. These models use neural networks with multiple layers to learn complex patterns in language.",
        "Training neural networks requires lots of data and computation. The process involves adjusting millions of parameters to minimize prediction errors.",
        "Transformers revolutionized natural language understanding. They use attention mechanisms to process sequences more effectively than previous architectures.",
        "Attention is all you need for sequence modeling. This phrase comes from the famous paper that introduced the Transformer architecture.",
        "GPT models are autoregressive language models. They generate text by predicting the next token based on all previous tokens in the sequence.",
        "Fine-tuning pretrained models is very effective. It allows us to adapt general language models to specific tasks with relatively little additional data.",
        "Large language models have emerged as powerful tools for various natural language tasks. They can perform translation, summarization, question answering, and text generation with remarkable accuracy.",
        "The field of artificial intelligence has grown rapidly in recent years. Advances in computing power and algorithmic improvements have enabled new breakthroughs in machine learning.",
        "Programming is both an art and a science. It requires logical thinking, creativity, and attention to detail to create efficient and maintainable software systems.",
        "Data science combines statistics, programming, and domain expertise. It involves collecting, cleaning, analyzing, and interpreting data to extract meaningful insights.",
        "Software engineering best practices include writing clean code, testing thoroughly, and documenting your work. These practices help create reliable and maintainable systems."
    ] * (num_samples // 15 + 1)  # Repeat to get enough samples
    
    def data_generator():
        for i in range(num_samples):
            text = training_texts[i % len(training_texts)]
            # Add some variation
            if i % 4 == 0:
                text = text.upper()
            elif i % 4 == 1:
                text = text.lower()
            elif i % 4 == 2:
                text = text.title()
            # else: keep original case
            
            # Tokenize
            tokens = tokenizer.encode(text)
            
            # Pad or truncate to seq_len
            if len(tokens) < seq_len:
                tokens.extend([tokenizer.char_to_id.get('<pad>', 0)] * (seq_len - len(tokens)))
            else:
                tokens = tokens[:seq_len]
            
            yield {'input_ids': np.array(tokens, dtype=np.int32)}
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature={
            'input_ids': tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
        }
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE), tokenizer

def create_improved_training_function(model, config):
    """Create improved training function with better loss handling"""
    
    # Create optimizer with your specified learning rate and gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,  # Using your 5e-4
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=1.0
    )
    
    # Loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, 
        reduction='none'
    )
    
    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    @tf.function
    def train_step(batch):
        input_ids = batch['input_ids']
        
        # Create inputs and targets
        inputs = input_ids[:, :-1]  # All except last token
        targets = input_ids[:, 1:]  # All except first token
        
        with tf.GradientTape() as tape:
            # Forward pass
            logits = model(inputs, training=True)
            
            # Ensure shapes match
            batch_size = tf.shape(targets)[0]
            seq_len = tf.shape(targets)[1]
            vocab_size = tf.shape(logits)[-1]
            
            # Reshape for loss computation
            targets_flat = tf.reshape(targets, [-1])
            logits_flat = tf.reshape(logits, [-1, vocab_size])
            
            # Create mask to ignore padding tokens
            mask = tf.cast(tf.not_equal(targets_flat, 0), tf.float32)  # 0 is pad token
            
            # Compute loss
            loss_per_token = loss_fn(targets_flat, logits_flat)
            masked_loss = loss_per_token * mask
            
            # Average over non-padded tokens
            total_loss = tf.reduce_sum(masked_loss)
            total_tokens = tf.reduce_sum(mask)
            loss = total_loss / tf.maximum(total_tokens, 1.0)
            
            # Add small regularization
            l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in model.trainable_variables 
                                   if 'bias' not in var.name and 'layer_norm' not in var.name])
            loss += 1e-6 * l2_loss
        
        # Compute and apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        train_loss.update_state(loss)
        train_accuracy.update_state(targets, logits, sample_weight=mask)
        
        return loss, tf.reduce_mean(tf.exp(tf.minimum(loss, 10.0)))  # Return loss and perplexity
    
    return train_step, optimizer, train_loss, train_accuracy

def improved_train_model(model, config):
    """Improved training function with better monitoring and debugging"""
    
    # Create dataset with your config parameters
    logger.info("Creating training dataset...")
    train_dataset, tokenizer = create_simple_text_dataset(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        num_samples=3000  # More samples for longer sequences
    )
    
    val_dataset, _ = create_simple_text_dataset(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        num_samples=300  # Validation samples
    )
    
    # Store tokenizer in model for later use
    model.tokenizer = tokenizer
    
    # Create training components
    train_step, optimizer, train_loss, train_accuracy = create_improved_training_function(model, config)
    
    # Create checkpoint directory
    checkpoint_dir = './improved_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training parameters - adjusted for larger model
    num_epochs = 10  # Fewer epochs for larger model
    best_loss = float('inf')
    patience_counter = 0
    patience = 8  # Reduced patience for larger model
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Reset metrics
        train_loss.reset_state()
        train_accuracy.reset_state()
        
        # Training
        batch_count = 0
        epoch_losses = []
        
        try:
            for batch in train_dataset:
                loss, perplexity = train_step(batch)
                batch_count += 1
                epoch_losses.append(float(loss))
                
                # Log every 25 batches (more frequent for batch_size=1)
                if batch_count % 25 == 0:
                    current_loss = float(train_loss.result())
                    current_acc = float(train_accuracy.result())
                    logger.info(
                        f"Epoch {epoch + 1}, Batch {batch_count}: "
                        f"Loss: {current_loss:.4f}, "
                        f"Accuracy: {current_acc:.4f}, "
                        f"Perplexity: {float(perplexity):.2f}"
                    )
        
        except Exception as e:
            logger.error(f"Error during training batch: {e}")
            continue
        
        # Epoch summary
        final_loss = float(train_loss.result())
        final_acc = float(train_accuracy.result())
        avg_perplexity = np.exp(np.minimum(np.mean(epoch_losses), 10.0))
        
        logger.info(
            f"Epoch {epoch + 1} Summary: "
            f"Loss: {final_loss:.4f}, "
            f"Accuracy: {final_acc:.4f}, "
            f"Perplexity: {avg_perplexity:.2f}"
        )
        
        # Early stopping check
        if final_loss < best_loss:
            best_loss = final_loss
            patience_counter = 0
            
            # Save best model
            try:
                best_model_path = os.path.join(checkpoint_dir, 'best_model.weights.h5')
                model.save_weights(best_model_path)
                logger.info(f"New best model saved: {best_model_path}")
            except Exception as e:
                logger.error(f"Error saving best model: {e}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Test generation every 5 epochs (more frequent for larger model)
        if (epoch + 1) % 5 == 0:
            try:
                test_generation(model, tokenizer, config)
            except Exception as e:
                logger.error(f"Error during test generation: {e}")
    
    # Load best model
    try:
        best_model_path = os.path.join(checkpoint_dir, 'best_model.weights.h5')
        if os.path.exists(best_model_path):
            model.load_weights(best_model_path)
            logger.info("Loaded best model weights")
    except Exception as e:
        logger.error(f"Error loading best model: {e}")
    
    return model

def test_generation(model, tokenizer, config, max_length=100):
    """Test text generation with the model"""
    try:
        # Test prompts
        test_prompts = [
            "Hello",
            "The quick",
            "Machine learning",
            "Natural language processing",
            "Programming is"
        ]
        
        logger.info("Testing text generation:")
        
        for prompt in test_prompts:
            # Encode prompt
            input_ids = tokenizer.encode(prompt)
            
            # Pad to minimum length
            if len(input_ids) < 20:
                input_ids.extend([0] * (20 - len(input_ids)))
            
            # Convert to tensor
            input_tensor = tf.constant([input_ids[:config.seq_len]], dtype=tf.int32)
            
            # Generate
            generated_ids = []
            current_input = input_tensor
            
            for _ in range(max_length):
                # Get logits (use seq_len-1 to match training)
                logits = model(current_input[:, :config.seq_len-1], training=False)
                
                # Get next token (greedy decoding)
                next_token = tf.argmax(logits[0, -1], axis=-1, output_type=tf.int32)
                generated_ids.append(int(next_token))
                
                # Update input (sliding window)
                new_input = tf.concat([current_input[:, 1:], [[next_token]]], axis=1)
                current_input = new_input
                
                # Stop if end token
                if int(next_token) == tokenizer.char_to_id.get('<eos>', 3):
                    break
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_ids)
            logger.info(f"Prompt: '{prompt}' -> Generated: '{generated_text}'")
            
    except Exception as e:
        logger.error(f"Error in test generation: {e}")

def debug_model_output(model, config):
    """Debug function to check model output shapes and values"""
    logger.info("Debugging model output...")
    
    try:
        # Create test input with your config
        batch_size = config.batch_size
        seq_len = config.seq_len - 1  # Match training setup
        test_input = tf.random.uniform(
            (batch_size, seq_len), 
            maxval=config.vocab_size, 
            dtype=tf.int32
        )
        
        logger.info(f"Test input shape: {test_input.shape}")
        logger.info(f"Test input values: {test_input[0, :5]}")
        
        # Forward pass
        output = model(test_input, training=False)
        
        logger.info(f"Model output shape: {output.shape}")
        logger.info(f"Expected output shape: ({batch_size}, {seq_len}, {config.vocab_size})")
        logger.info(f"Output logits range: [{tf.reduce_min(output):.4f}, {tf.reduce_max(output):.4f}]")
        logger.info(f"Output mean: {tf.reduce_mean(output):.4f}")
        logger.info(f"Output std: {tf.math.reduce_std(output):.4f}")
        
        # Check if model is outputting reasonable probabilities
        probs = tf.nn.softmax(output, axis=-1)
        logger.info(f"Max probability: {tf.reduce_max(probs):.4f}")
        logger.info(f"Min probability: {tf.reduce_min(probs):.4f}")
        
        # Check if all outputs are the same (indicating no learning)
        if tf.reduce_std(output) < 1e-6:
            logger.warning("Model outputs have very low variance - model may not be learning!")
        
    except Exception as e:
        logger.error(f"Error in model debugging: {e}")

# Modified main execution with your configuration
if __name__ == "__main__":
    # Import your model configuration and class
    # from minigpt_transformer import EnhancedMiniGPT, ModelConfig
    
    # Create model configuration with your specified settings
    config_dict = {
        'vocab_size': 50257,
        'max_seq_len': 512,
        'embed_dim': 384,
        'num_heads': 6,
        'num_layers': 6,
        'ffn_dim': 1536,
        'dropout': 0.1,
        'use_custom_attention': True,
        'use_rotary_embeddings': True,
        'learning_rate': 5e-4,
        'batch_size': 1,
        'seq_len': 512
    }
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    config = SimpleConfig(**config_dict)
    
    logger.info("Configuration:")
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")
    
    # Calculate approximate model parameters
    embed_params = config.vocab_size * config.embed_dim
    layer_params = config.num_layers * (
        # Multi-head attention
        4 * config.embed_dim * config.embed_dim +  # Q, K, V, O projections
        # Feed-forward network
        2 * config.embed_dim * config.ffn_dim +
        # Layer norms (approximate)
        4 * config.embed_dim
    )
    total_params = embed_params + layer_params
    logger.info(f"Estimated model parameters: {total_params:,}")
    
    # If you have the model available, uncomment these lines:
    # logger.info("Creating model...")
    # model = EnhancedMiniGPT(config)
    # 
    # # Build model
    # dummy_input = tf.random.uniform((1, config.seq_len-1), maxval=config.vocab_size, dtype=tf.int32)
    # _ = model(dummy_input)
    # actual_params = model.count_params()
    # logger.info(f"Model created with {actual_params:,} parameters")
    # 
    # # Debug model
    # debug_model_output(model, config)
    # 
    # # Train model
    # logger.info("Starting improved training...")
    # model = improved_train_model(model, config)
    # 
    # logger.info("Training completed!")
    
    logger.info("Improved training script ready with your configuration. Uncomment the model creation lines to run.")