import tensorflow as tf
from datetime import datetime
import json
from pathlib import Path
import re
import logging
import numpy as np
import os
from datasets import load_dataset
import random

# Import from the model file
try:
    from transformers import AutoTokenizer  # Fixed: GPT2tokenizer -> AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: transformers library not found. Please install with: pip install transformers")
    HAS_TRANSFORMERS = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetLoader:
    """Enhanced dataset loader with GPT-2 tokenizer"""
    
    def __init__(self, seq_len=1024):
        # Initialize GPT-2 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Initialized GPT-2 tokenizer")
        except Exception as e:
            logger.error(f"Failed to initialize GPT-2 tokenizer: {e}")
            raise
            
        self.seq_len = seq_len
        self.datasets = {}
        
    def load_daily_dialogue(self, split='train', max_samples=None):
        """Load and process the daily_dialogue dataset"""
        try:
            logger.info(f"Loading daily_dialogue dataset ({split} split)...")
            dataset = load_dataset("daily_dialog", split=split)
            
            if max_samples:
                dataset = dataset.select(range(min(len(dataset), max_samples)))
            
            processed_texts = []
            for example in dataset:
                # Join dialogue turns with speaker tokens
                dialogue = example['dialog']
                processed_dialogue = []
                
                for i, turn in enumerate(dialogue):
                    speaker = "Human" if i % 2 == 0 else "Assistant"
                    processed_dialogue.append(f"{speaker}: {turn}")
                
                # Join all turns in the dialogue
                full_dialogue = "\n".join(processed_dialogue)
                processed_texts.append(full_dialogue)
            
            logger.info(f"Loaded {len(processed_texts)} dialogues from daily_dialogue")
            return processed_texts
            
        except Exception as e:
            logger.error(f"Error loading daily_dialogue: {e}")
            return []
    
    def load_conv2ai(self, split='train', max_samples=None):
        """Load and process alternative conversation datasets"""
        try:
            logger.info(f"Attempting to load conversation dataset ({split} split)...")
            
            # Try multiple potential dataset names
            dataset_names = ["conv_ai_2", "blended_skill_talk", "empathetic_dialogues"]
            dataset = None
            
            for name in dataset_names:
                try:
                    dataset = load_dataset(name, split=split)
                    logger.info(f"Successfully loaded {name} dataset")
                    break
                except Exception as e:
                    logger.warning(f"Could not load {name}: {e}")
                    continue
            
            if dataset is None:
                raise Exception("No conversation datasets available")
            
            if max_samples:
                dataset = dataset.select(range(min(len(dataset), max_samples)))
            
            processed_texts = []
            for example in dataset:
                # Handle different dataset structures
                if 'text' in example:
                    processed_texts.append(example['text'])
                elif 'dialogue' in example:
                    if isinstance(example['dialogue'], list):
                        dialogue_text = "\n".join([f"Turn {i+1}: {turn}" for i, turn in enumerate(example['dialogue'])])
                    else:
                        dialogue_text = str(example['dialogue'])
                    processed_texts.append(dialogue_text)
                elif 'utterances' in example:
                    utterances = example['utterances']
                    if isinstance(utterances, list):
                        dialogue_text = "\n".join([f"Speaker: {utt}" for utt in utterances])
                    else:
                        dialogue_text = str(utterances)
                    processed_texts.append(dialogue_text)
                else:
                    # Fallback - convert entire example to string
                    processed_texts.append(str(example))
            
            logger.info(f"Loaded {len(processed_texts)} conversations from dataset")
            return processed_texts
            
        except Exception as e:
            logger.error(f"Error loading conversation datasets: {e}")
            logger.info("Using fallback synthetic conversation data")
            return self._create_synthetic_conversations(max_samples or 1000)
    
    def _create_synthetic_conversations(self, num_samples=1000):
        """Create synthetic conversation data as fallback"""
        conversation_templates = [
            "Human: How are you today?\nI'm doing well, thank you! How about you?\nHuman: I'm great, thanks for asking.",
            "Human: What's the weather like?\nI don't have access to current weather data, but I'd be happy to help you find weather information.\nHuman: That's okay, I'll check online.",
            "Human: Can you help me with a programming question?\nOf course! I'd be happy to help with programming. What's your question?\nHuman: I need to understand how loops work in Python.",
            "Human: Tell me about machine learning.\nMachine learning is a subset of AI that enables computers to learn from data without explicit programming.\nHuman: That sounds interesting. Can you give me an example?",
            "Human: I'm feeling stressed about work.\nI understand work stress can be challenging. Would you like to talk about what's causing the stress?\nHuman: Yes, I have too many deadlines coming up.",
            "Human: What's your favorite programming language?\nI don't have personal preferences, but I can help you choose a language based on your needs.\nHuman: I want to learn web development.",
            "Human: Explain quantum computing.\nQuantum computing uses quantum mechanical phenomena to process information in ways classical computers cannot.\nHuman: That sounds complex but fascinating.",
            "Human: How do neural networks work?\nNeural networks are inspired by the human brain and consist of interconnected nodes that process information.\nHuman: Can you give me a simple example?",
            "Human: What should I learn first in data science?\nI'd recommend starting with Python programming and basic statistics.\nHuman: Where can I find good resources for learning?",
            "Human: I'm having trouble debugging my code.\nI'd be happy to help! Can you tell me what error you're encountering?\nHuman: I keep getting a syntax error but I can't find it."
        ]
        
        synthetic_conversations = []
        for i in range(num_samples):
            template = random.choice(conversation_templates)
            # Add some variation
            if i % 3 == 0:
                template = template.replace("Human:", "User:")
            elif i % 3 == 1:
                template = template.replace("I'm", "I am").replace("I'd", "I would").replace("I'll", "I will")
            
            synthetic_conversations.append(template)
        
        return synthetic_conversations
    
    def create_combined_dataset(self, daily_dialogue_samples=2000, conv2ai_samples=1000, 
                              include_synthetic=True, synthetic_samples=1000):
        """Create a combined dataset from multiple sources"""
        all_texts = []
        
        # Load daily_dialogue
        daily_texts = self.load_daily_dialogue(max_samples=daily_dialogue_samples)
        all_texts.extend(daily_texts)
        
        # Load alternative conversation datasets
        conv2ai_texts = self.load_conv2ai(max_samples=conv2ai_samples)
        all_texts.extend(conv2ai_texts)
        
        # Add synthetic data if requested
        if include_synthetic:
            synthetic_texts = self._create_synthetic_conversations(synthetic_samples)
            all_texts.extend(synthetic_texts)
        
        # Add some general text data for diversity
        general_texts = self._create_general_text_samples(500)
        all_texts.extend(general_texts)
        
        logger.info(f"Created combined dataset with {len(all_texts)} samples")
        return all_texts
    
    def _create_general_text_samples(self, num_samples=500):
        """Create general text samples for training diversity"""
        general_texts = [
            "The importance of artificial intelligence in modern technology cannot be overstated. I have applications in healthcare, finance, transportation, and many other sectors.",
            "Natural language processing enables computers to understand and generate human language. I power chatbots, translation services, and text analysis tools.",
            "Deep learning models use neural networks with multiple layers to learn complex patterns in data. I have achieved remarkable success in image recognition, speech processing, and language understanding.",
            "Python is a versatile programming language used for web development, data science, machine learning, and automation. My simple syntax makes me popular among beginners and experts alike.",
            "Cloud computing has revolutionized how businesses store and process data. I offer scalability, cost-effectiveness, and accessibility from anywhere in the world.",
            "Cybersecurity is crucial in our digital age. Organizations must protect their data and systems from various threats including malware, phishing, and unauthorized access. I help ensure digital security.",
            "The Internet of Things (IoT) connects everyday devices to the internet, enabling smart homes, wearable technology, and industrial automation. I facilitate these connections.",
            "Blockchain technology provides a secure and transparent way to record transactions. I have applications beyond cryptocurrency, including supply chain management and digital identity verification.",
            "Data visualization helps people understand complex information through charts, graphs, and interactive dashboards. I am essential for business intelligence and scientific research.",
            "Software engineering best practices include version control, code review, testing, and documentation. I ensure code quality and maintainability through these practices."
        ] * (num_samples // 10 + 1)
        
        return general_texts[:num_samples]

def create_enhanced_dataset(seq_len=1024, batch_size=1, 
                          daily_dialogue_samples=2000, conv2ai_samples=1000,
                          include_synthetic=True, validation_split=0.1):
    """Create enhanced dataset with GPT-2 tokenizer - FIXED"""
    
    # Initialize dataset loader (tokenizer is initialized inside)
    loader = DatasetLoader(seq_len=seq_len)
    tokenizer = loader.tokenizer  # Get tokenizer from loader
    
    # Create combined dataset
    all_texts = loader.create_combined_dataset(
        daily_dialogue_samples=daily_dialogue_samples,
        conv2ai_samples=conv2ai_samples,
        include_synthetic=include_synthetic
    )
    
    # Shuffle the data
    random.shuffle(all_texts)
    
    # Split into train and validation
    split_idx = int(len(all_texts) * (1 - validation_split))
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    logger.info(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    
    # Create TensorFlow datasets
    train_dataset = create_tf_dataset_from_texts(train_texts, tokenizer, seq_len, batch_size)
    val_dataset = create_tf_dataset_from_texts(val_texts, tokenizer, seq_len, batch_size)
    
    return train_dataset, val_dataset, tokenizer

def create_tf_dataset_from_texts(texts, tokenizer, seq_len, batch_size):
    """Convert text list to TensorFlow dataset - FIXED"""
    
    def text_generator():
        for text in texts:
            # Tokenize with proper handling
            try:
                # Encode text with padding and truncation
                encoded = tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=seq_len + 1,
                    truncation=True,
                    padding='max_length',
                )
                
                # Ensure we have the right length
                if len(encoded) < seq_len + 1:
                    encoded = encoded + [tokenizer.pad_token_id] * (seq_len + 1 - len(encoded))
                elif len(encoded) > seq_len + 1:
                    encoded = encoded[:seq_len + 1]
                
                encoded = np.array(encoded, dtype=np.int32)
                
                # Create input/target pairs for causal language modeling
                input_ids = encoded[:-1]
                target_ids = encoded[1:]
                
                yield {
                    'input_ids': input_ids,
                    'target_ids': target_ids
                }
                
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                # Skip problematic texts
                continue
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_generator(
        text_generator,
        output_signature={
            'input_ids': tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            'target_ids': tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
        }
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_improved_training_function(model, config):
    """Creates an improved training function for the MoEMiniGPT model."""
    
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
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    
    @tf.function
    def train_step(batch):
        inputs = batch['input_ids']
        targets = batch['target_ids']
        
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                targets, predictions, from_logits=True
            )
            loss = tf.reduce_mean(loss)
            
            # Add MoE auxiliary losses (they're automatically added via model.losses)
            total_loss = loss + tf.reduce_sum(model.losses)
        
        # Compute and apply gradients with clipping
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        train_loss.update_state(total_loss)
        train_accuracy.update_state(targets, predictions)
        
        return total_loss
    
    @tf.function
    def val_step(batch):
        inputs = batch['input_ids']
        targets = batch['target_ids']
        
        predictions = model(inputs, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            targets, predictions, from_logits=True
        )
        loss = tf.reduce_mean(loss)
        
        # Add MoE auxiliary losses
        total_loss = loss + tf.reduce_sum(model.losses)
        
        # Update validation metrics
        val_loss.update_state(total_loss)
        val_accuracy.update_state(targets, predictions)
        
        return total_loss
    
    return train_step, val_step, optimizer, (train_loss, train_accuracy, val_loss, val_accuracy)

def enhanced_train_model(model, config):
    """Enhanced training function with dataset integration and improved monitoring - FIXED"""
    
    # Create enhanced datasets
    logger.info("Creating enhanced training datasets...")
    try:
        # Create datasets (tokenizer is handled inside)
        train_dataset, val_dataset, tokenizer = create_enhanced_dataset(
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            daily_dialogue_samples=1500,  # Reduced for faster training
            conv2ai_samples=800,
            include_synthetic=True
        )
        
        # Store tokenizer in model
        model._tokenizer = tokenizer
        
    except Exception as e:
        logger.error(f"Error creating enhanced datasets: {e}")
        logger.info("Falling back to simple text dataset...")
        train_dataset, tokenizer = create_simple_text_dataset(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            num_samples=2000
        )
        val_dataset, _ = create_simple_text_dataset(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            num_samples=200
        )
        model._tokenizer = tokenizer
    
    # Create training components
    train_step, val_step, optimizer, metrics = create_improved_training_function(model, config)
    train_loss, train_accuracy, val_loss, val_accuracy = metrics
    
    # Create checkpoint directory
    checkpoint_dir = './enhanced_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training parameters
    num_epochs = 15
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Reset metrics
        for metric in metrics:
            metric.reset_state()
        
        # Training phase
        batch_count = 0
        try:
            for batch in train_dataset:
                loss = train_step(batch)
                batch_count += 1
                
                # Log progress every 50 batches
                if batch_count % 50 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}, Batch {batch_count}: "
                        f"Loss: {float(train_loss.result()):.4f}, "
                        f"Accuracy: {float(train_accuracy.result()):.4f}"
                    )
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            continue
        
        # Validation phase
        val_batch_count = 0
        try:
            for batch in val_dataset:
                val_step(batch)
                val_batch_count += 1
                
        except Exception as e:
            logger.error(f"Error during validation: {e}")
        
        # Epoch summary
        epoch_train_loss = float(train_loss.result())
        epoch_train_acc = float(train_accuracy.result())
        epoch_val_loss = float(val_loss.result()) if val_batch_count > 0 else float('inf')
        epoch_val_acc = float(val_accuracy.result()) if val_batch_count > 0 else 0.0
        
        # Store history
        history['train_loss'].append(epoch_train_loss)
        history['train_accuracy'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)
        
        logger.info(
            f"Epoch {epoch + 1} Summary: "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Train Acc: {epoch_train_acc:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val Acc: {epoch_val_acc:.4f}"
        )
        
        # Early stopping and model saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
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
        
        # Test generation every 3 epochs
        if (epoch + 1) % 3 == 0:
            try:
                test_generation(model, model._tokenizer, config)
            except Exception as e:
                logger.error(f"Error during test generation: {e}")
        
        # Save training history
        try:
            history_path = os.path.join(checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
    
    # Load best model
    try:
        best_model_path = os.path.join(checkpoint_dir, 'best_model.weights.h5')
        if os.path.exists(best_model_path):
            model.load_weights(best_model_path)
            logger.info("Loaded best model weights")
    except Exception as e:
        logger.error(f"Error loading best model: {e}")
    
    return model, history

def test_generation(model, tokenizer, config):
    """Test text generation with the MoEMiniGPT model - FIXED"""
    try:
        # FIXED: Define max_length
        max_length = 50
        
        # Enhanced test prompts for dialogue and conversation
        test_prompts = [
            "Human: Hello, how are you today?",
            "User: Can you help me with a question?",
            "Human: What's the weather like?",
            "I'd be happy to help you with",
            "The conversation started when",
            "In machine learning,",
            "Programming is",
            "Human: I'm feeling stressed about work.\n"
        ]
        
        logger.info("Testing enhanced text generation:")
        for prompt in test_prompts:
            try:
                # Encode prompt
                input_ids = tokenizer.encode(prompt, add_special_tokens=True)
                
                # Ensure input doesn't exceed sequence length
                if len(input_ids) >= config.seq_len:
                    input_ids = input_ids[:config.seq_len-10]  # Leave room for generation
                
                # Pad input to match model expectations
                input_array = np.zeros(config.seq_len, dtype=np.int32)
                input_array[:len(input_ids)] = input_ids
                
                # Convert to tensor
                current_input = tf.convert_to_tensor(
                    input_array[None, :],
                    dtype=tf.int32
                )
                
                # Generate with improved sampling
                generated_ids = []
                current_length = len(input_ids)
                
                for _ in range(max_length):
                    # Ensure we don't exceed sequence length
                    if current_length >= config.seq_len - 1:
                        break
                        
                    # Get model predictions
                    logits = model(current_input, training=False)
                    next_token_logits = logits[0, current_length - 1, :]  # Use current position
                    
                    # Apply temperature and top-k sampling
                    temperature = 0.8
                    top_k = 50
                    
                    # Temperature scaling
                    next_token_logits = next_token_logits / temperature
                    
                    # Top-k filtering
                    top_k_logits, top_k_indices = tf.nn.top_k(next_token_logits, k=top_k)
                    next_token_logits = tf.where(
                        next_token_logits < top_k_logits[-1],
                        tf.ones_like(next_token_logits) * -1e9,
                        next_token_logits
                    )
                    
                    # Sample next token
                    probs = tf.nn.softmax(next_token_logits, axis=-1)
                    next_token = tf.random.categorical(tf.math.log(probs[None, :] + 1e-10), num_samples=1)[0, 0]
                    
                    token_id = int(next_token.numpy())
                    generated_ids.append(token_id)
                    
                    # Update input array
                    if current_length < config.seq_len:
                        input_array[current_length] = token_id
                        current_length += 1
                    
                    # Update input tensor (shift left and add new token)
                    current_input = tf.convert_to_tensor(input_array[None, :], dtype=tf.int32)
                    
                    # Stop at end token or newline for dialogue
                    if token_id == tokenizer.eos_token_id:
                        break
                
                # Decode and display
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                logger.info(f"Prompt: '{prompt}'")
                logger.info(f"Generated: '{generated_text}'")
                logger.info("-" * 50)
                
            except Exception as e:
                logger.error(f"Error generating for prompt '{prompt}': {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error in enhanced test generation: {e}")

def create_simple_text_dataset(vocab_size=50257, seq_len=1024, batch_size=1, num_samples=1000):
    """Fallback simple text dataset creation - FIXED"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Using GPT-2 tokenizer for simple dataset")
    except ImportError:
        logger.error("Could not import transformers. Please install with: pip install transformers")
        raise
    
    # Create diverse training texts (enhanced version)
    training_texts = [
        "Hello world! How are you today? I hope you are doing well and having a great time learning about natural language processing.",
        "Human: How can I help you today?\nI'm here to help with any questions you might have. What would you like to know?",
        "The conversation between humans and AI systems is becoming increasingly natural and helpful. I facilitate these interactions.",
        "Machine learning enables computers to learn patterns from data and make intelligent decisions. I am an example of this technology.",
        "Natural language processing allows computers to understand and generate human language effectively. I use these techniques to communicate.",
        "Deep learning models use neural networks to process complex patterns in text and speech. I am built using these advanced methods.",
        "Training language models requires large datasets and significant computational resources. I was trained using these approaches.",
        "Transformers have revolutionized the field of natural language understanding and generation. I am based on transformer architecture.",
        "Human: What is artificial intelligence?\nArtificial intelligence is the simulation of human intelligence in machines. I am an example of AI.",
        "The development of conversational AI has made significant progress in recent years. I represent this advancement.",
        "User: Can you explain machine learning?\nMachine learning is a subset of AI that learns from data. I use machine learning techniques.",
        "Programming languages like Python are essential for developing AI applications. I was developed using these tools.",
        "Data science combines statistics, programming, and domain knowledge to extract insights. I apply these principles.",
        "Human: I'm interested in learning about neural networks.\nNeural networks are inspired by the human brain. I am built using neural network architectures.",
        "The future of AI includes applications in healthcare, education, and many other fields. I can contribute to these areas."
    ] * (num_samples // 15 + 1)
    
    def data_generator():
        for i in range(num_samples):
            text = training_texts[i % len(training_texts)]
            
            # Add variations
            if i % 4 == 0:
                text = text.upper()
            elif i % 4 == 1:
                text = text.lower()
            elif i % 4 == 2:
                text = text.title()
            
            try:
                # Tokenize - FIXED
                encoded = tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=seq_len + 1,
                    truncation=True,
                    padding='max_length',
                )
                
                # Ensure we have the right length
                if len(encoded) < seq_len + 1:
                    encoded = encoded + [tokenizer.pad_token_id] * (seq_len + 1 - len(encoded))
                elif len(encoded) > seq_len + 1:
                    encoded = encoded[:seq_len + 1]
                
                encoded = np.array(encoded, dtype=np.int32)
                
                # Create input/target pairs
                input_ids = encoded[:-1]
                target_ids = encoded[1:]
                
                yield {
                    'input_ids': input_ids,
                    'target_ids': target_ids
                }
            except Exception as e:
                logger.warning(f"Error processing text in simple dataset: {e}")
                continue
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature={
            'input_ids': tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            'target_ids': tf.TensorSpec(shape=(seq_len,), dtype=tf.int32)
        }
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE), tokenizer

# Example usage and main execution - FIXED
if __name__ == "__main__":
    logger.info("Starting MoEMiniGPT training script...")
    
    try:
        # Import the MoEMiniGPT and MoEConfig from minigpt_transformer
        from minigpt_transformer import MoEMiniGPT, MoEConfig
        
        # Create configuration
        config = MoEConfig(
            vocab_size=50257,
            max_seq_len=1024,
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            ffn_dim=3072,
            dropout=0.1,
            layer_norm_epsilon=1e-5,
            use_rotary_embeddings=True,
            learning_rate=1e-4,
            batch_size=1,
            seq_len=1024,
            num_experts=8,
            top_k_experts=2,
            use_moe_layers=[2, 4, 6, 8, 10]  # Use MoE in these layers
        )
        
        # Create model
        logger.info("Initializing MoEMiniGPT model...")
        model = MoEMiniGPT(config)
        
        # Initialize model with dummy input
        dummy_input = tf.random.uniform((1, config.seq_len), maxval=config.vocab_size, dtype=tf.int32)
        _ = model(dummy_input)
        
        # Log model summary
        total_params = model.count_params()
        logger.info(f"Model initialized with {total_params:,} parameters")
        
        # Train model
        logger.info("Starting model training...")
        trained_model, history = enhanced_train_model(model, config)
        
        # Save final model and configuration
        save_dir = "trained_models"
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Save model weights
            model_path = os.path.join(save_dir, "moe_minigpt_final.h5")
            trained_model.save_weights(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save configuration
            config_path = os.path.join(save_dir, "moe_config.json")
            with open(config_path, 'w') as f:
                # Convert config to dictionary, handling special types
                config_dict = {k: str(v) if isinstance(v, (list, type(None))) else v 
                             for k, v in vars(config).items()}
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
            
            # Save training history
            history_path = os.path.join(save_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Training history saved to {history_path}")
            
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
        
        # Final generation test
        logger.info("\nRunning final generation tests...")
        test_prompts = [
            "Human: What can you tell me about machine learning?",
            "Human: How does a transformer model work?",
            "Human: Write a short poem about AI."
        ]
        
        for prompt in test_prompts:
            try:
                generated_text = trained_model.generate_text(
                    prompt=prompt,
                    max_length=100,
                    temperature=0.7
                )
                logger.info(f"\nPrompt: {prompt}")
                logger.info(f"Generated: {generated_text}")
            except Exception as e:
                logger.error(f"Generation error for prompt '{prompt}': {e}")
        
        # Print final statistics
        logger.info("\nTraining completed successfully!")
        logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise