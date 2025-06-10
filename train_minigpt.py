import tensorflow as tf
import numpy as np
import pickle
import os
from datetime import datetime
import json
import logging
import datasets
from datasets import disable_caching
from minigpt_transformer import MiniGPT, ChatTokenizer

# Disable datasets caching to prevent disk space issues
disable_caching()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure GPU/CPU settings
def setup_gpu():
    """Setup GPU configuration with proper error handling"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU configured: {len(gpus)} GPU(s) available")
            return True
        else:
            logger.info("No GPU found, using CPU")
            return False
    except Exception as e:
        logger.warning(f"GPU configuration failed: {e}")
        logger.info("Falling back to CPU")
        return False

# Training configuration for large model
TRAINING_CONFIG = {
    'batch_size': 32,  # Reduced for large model
    'num_epochs': 5,   # Reduced for large datasets
    'learning_rate': 0.00005,  # Lower learning rate for stability
    'max_seq_len': 512,  # Increased for better context
    'target_tokens': 10000000,  # 10M tokens target
    'max_samples_per_dataset': 100000,  # Limit per dataset to manage memory
}

def safe_dataset_load(dataset_name, config_name=None, split="train", max_samples=None):
    """Safely load dataset with proper error handling"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        if config_name:
            dataset = datasets.load_dataset(dataset_name, config_name, split=split, cache_dir=cache_dir)
        else:
            dataset = datasets.load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        
        # Limit dataset size if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            
        return dataset
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return None

def extract_conversations_from_dataset(dataset, dataset_name):
    """Extract conversations from different dataset formats"""
    texts = []
    
    try:
        if dataset is None:
            return texts
            
        logger.info(f"Processing {dataset_name} with {len(dataset)} examples...")
        
        for idx, example in enumerate(dataset):
            try:
                # Different extraction methods based on dataset structure
                if 'dialog' in example or 'dialogue' in example:
                    # Handle dialog datasets
                    dialog_key = 'dialog' if 'dialog' in example else 'dialogue'
                    dialog = example[dialog_key]
                    
                    if isinstance(dialog, list):
                        for i in range(len(dialog) - 1):
                            if isinstance(dialog[i], dict):
                                q_text = dialog[i].get('text', '').strip()
                                a_text = dialog[i+1].get('text', '').strip() if i+1 < len(dialog) else ''
                            else:
                                q_text = str(dialog[i]).strip()
                                a_text = str(dialog[i+1]).strip() if i+1 < len(dialog) else ''
                            
                            if q_text and a_text and len(q_text) > 2 and len(a_text) > 2:
                                texts.append(f"Q: {q_text}")
                                texts.append(f"A: {a_text}")
                
                elif 'conversations' in example:
                    # Handle conversation datasets like ShareGPT
                    convs = example['conversations']
                    if isinstance(convs, list) and len(convs) >= 2:
                        for i in range(0, len(convs)-1, 2):
                            if i+1 < len(convs):
                                q_text = str(convs[i]).strip()
                                a_text = str(convs[i+1]).strip()
                                if q_text and a_text:
                                    texts.append(f"Q: {q_text}")
                                    texts.append(f"A: {a_text}")
                
                elif 'utterance' in example and 'context' in example:
                    # Handle empathetic dialogues
                    context = example['context'].strip()
                    utterance = example['utterance'].strip()
                    if context and utterance:
                        texts.append(f"Q: {context}")
                        texts.append(f"A: {utterance}")
                
                elif 'prompt' in example and 'response' in example:
                    # Handle prompt-response datasets
                    prompt = example['prompt'].strip()
                    response = example['response'].strip()
                    if prompt and response:
                        texts.append(f"Q: {prompt}")
                        texts.append(f"A: {response}")
                
                elif 'input' in example and 'output' in example:
                    # Handle input-output datasets
                    inp = example['input'].strip()
                    out = example['output'].strip()
                    if inp and out:
                        texts.append(f"Q: {inp}")
                        texts.append(f"A: {out}")
                
                elif 'text' in example and 'summary' in example:
                    # Handle text-summary datasets
                    text = example['text'].strip()
                    summary = example['summary'].strip()
                    if text and summary:
                        texts.append(f"Q: Summarize this: {text}")
                        texts.append(f"A: {summary}")
                
                elif 'question' in example and 'answer' in example:
                    # Handle Q&A datasets
                    question = example['question'].strip()
                    answer = example['answer'].strip()
                    if question and answer:
                        texts.append(f"Q: {question}")
                        texts.append(f"A: {answer}")
                
                # Progress logging
                if (idx + 1) % 10000 == 0:
                    logger.info(f"Processed {idx + 1} examples from {dataset_name}")
                    
            except Exception as e:
                logger.warning(f"Error processing example {idx} from {dataset_name}: {e}")
                continue
        
        logger.info(f"Extracted {len(texts)} texts from {dataset_name}")
        return texts
        
    except Exception as e:
        logger.error(f"Failed to extract from {dataset_name}: {e}")
        return []

def load_all_conversation_datasets():
    """Load all available conversation datasets"""
    logger.info("Loading conversation datasets...")
    all_texts = []
    total_tokens = 0
    
    # Define datasets to try loading
    datasets_config = [
        # Core conversation datasets
        ("daily_dialog", None),           # DailyDialog
        ("empathetic_dialogues", None),   # Empathetic Dialogues
        ("conv_ai_2", None),             # ConvAI2
        ("PygmalionAI/PIPPA", None),     # PIPPA
        ("OpenAssistant/oasst1", None),  # OpenAssistant Conversations
        ("HuggingFaceH4/ultrachat_200k", None),  # GPT-4/GPT-3.5 Augmented
    ]
    
    for dataset_name, config_name in datasets_config:
        try:
            logger.info(f"Attempting to load {dataset_name}" + (f" ({config_name})" if config_name else ""))
            
            dataset = safe_dataset_load(
                dataset_name, 
                config_name, 
                max_samples=TRAINING_CONFIG['max_samples_per_dataset']
            )
            
            if dataset is not None:
                texts = extract_conversations_from_dataset(dataset, dataset_name)
                if texts:
                    all_texts.extend(texts)
                    
                    # Calculate tokens and check if we've reached target
                    current_tokens = sum(len(text.split()) for text in texts)
                    total_tokens += current_tokens
                    
                    logger.info(f"Added {len(texts):,} texts ({current_tokens:,} tokens) from {dataset_name}")
                    logger.info(f"Total: {len(all_texts):,} texts, {total_tokens:,} tokens")
                    
                    # Check if we've reached our token target
                    if total_tokens >= TRAINING_CONFIG['target_tokens']:
                        logger.info(f"Reached target of {TRAINING_CONFIG['target_tokens']:,} tokens")
                        break
            
            # Clean up memory
            del dataset
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")
            continue
    
    # Add fallback conversation data if we don't have enough
    if len(all_texts) < 1000:
        logger.warning("Adding fallback conversation data...")
        fallback_texts = create_fallback_conversation_data()
        all_texts.extend(fallback_texts)
        total_tokens += sum(len(text.split()) for text in fallback_texts)
    
    logger.info(f"Final dataset: {len(all_texts):,} texts, {total_tokens:,} tokens")
    return all_texts

def create_fallback_conversation_data():
    """Create fallback conversation data"""
    logger.info("Creating fallback conversation data...")
    
    conversation_patterns = [
        ("Hello", "Hi there! How can I help you today?"),
        ("Hi", "Hello! Nice to meet you. What's on your mind?"),
        ("How are you?", "I'm doing well, thank you for asking! How about you?"),
        ("What's your name?", "I'm an AI assistant. You can call me Assistant."),
        ("Can you help me?", "Absolutely! I'm here to help. What do you need assistance with?"),
        ("Thank you", "You're very welcome! Is there anything else I can help you with?"),
        ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
        ("What's the weather like?", "I don't have access to current weather data, but you can check a weather app."),
        ("What time is it?", "I don't have access to the current time. Please check your device's clock."),
        ("Goodbye", "Goodbye! It was wonderful chatting with you. Take care!"),
        ("Good morning", "Good morning! I hope you're having a great day so far."),
        ("Good night", "Good night! Sleep well and sweet dreams."),
        ("How old are you?", "I'm an AI, so I don't have an age in the traditional sense."),
        ("Where are you from?", "I exist in the digital realm, created by my developers."),
        ("What can you do?", "I can help answer questions, have conversations, and assist with various tasks."),
        ("I'm sad", "I'm sorry to hear you're feeling sad. Would you like to talk about what's bothering you?"),
        ("I'm happy", "That's wonderful to hear! I'm glad you're feeling happy today."),
        ("Tell me about yourself", "I'm an AI assistant designed to be helpful, harmless, and honest."),
        ("What's your favorite color?", "I don't have personal preferences, but I find all colors fascinating in their own way."),
        ("Do you like music?", "I think music is a beautiful form of human expression and creativity."),
    ]
    
    texts = []
    # Create substantial fallback data
    for _ in range(1000):
        for q, a in conversation_patterns:
            texts.append(f"Q: {q}")
            texts.append(f"A: {a}")
    
    logger.info(f"Created {len(texts)} fallback texts")
    return texts

def create_training_sequences(texts, tokenizer, seq_len, batch_size=1000):
    """Create training sequences in batches to manage memory"""
    logger.info("Creating training sequences...")
    
    inputs, targets = [], []
    pad_token_id = 0
    total_tokens = 0
    
    # Process in batches to manage memory
    for batch_start in range(0, len(texts), batch_size * 2):  # *2 because we process Q&A pairs
        batch_end = min(batch_start + batch_size * 2, len(texts))
        batch_texts = texts[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(texts) + batch_size*2 - 1)//(batch_size*2)}")
        
        for i in range(0, len(batch_texts)-1, 2):
            if i+1 < len(batch_texts):
                try:
                    # Get question and answer
                    question = batch_texts[i].replace("Q: ", "").strip()
                    answer = batch_texts[i+1].replace("A: ", "").strip()
                    
                    # Skip very short or very long sequences
                    if len(question) < 3 or len(answer) < 3:
                        continue
                    if len(question) > 1000 or len(answer) > 1000:
                        continue
                    
                    # Combine into training text
                    combined_text = f"{question} {answer}"
                    
                    # Tokenize
                    sequence = tokenizer.texts_to_sequences([combined_text])[0]
                    
                    if len(sequence) > 1:
                        total_tokens += len(sequence)
                        
                        # Pad or truncate to seq_len + 1
                        if len(sequence) <= seq_len:
                            padded = sequence + [pad_token_id] * (seq_len + 1 - len(sequence))
                        else:
                            padded = sequence[:seq_len + 1]
                        
                        # Create input-target pairs (shift by 1)
                        inputs.append(padded[:-1])
                        targets.append(padded[1:])
                        
                except Exception as e:
                    logger.warning(f"Error processing sequence: {e}")
                    continue
    
    logger.info(f"Created {len(inputs):,} training sequences with {total_tokens:,} tokens")
    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """Masked loss function for padded sequences"""
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    masked_loss = loss * mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-8)

def masked_accuracy(y_true, y_pred):
    """Masked accuracy for padded sequences"""
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    correct = tf.cast(tf.equal(y_true, predictions), tf.float32) * mask
    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-8)

def train_model():
    """Main training function for large model"""
    logger.info("=== STARTING LARGE MINIGPT TRAINING ===")
    
    try:
        # Setup hardware
        gpu_available = setup_gpu()
        
        # Set TensorFlow memory management
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_intra_op_parallelism_threads(0)
        
        # Load datasets
        logger.info("Loading conversation datasets...")
        texts = load_all_conversation_datasets()
        
        if len(texts) == 0:
            raise ValueError("No training data loaded")
        
        # Initialize tokenizer
        logger.info("Creating and fitting tokenizer...")
        tokenizer = ChatTokenizer()
        
        # Fit tokenizer in batches to manage memory
        batch_size = 10000
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            if i == 0:
                tokenizer.fit_on_texts(batch_texts)
            else:
                tokenizer.update_on_texts(batch_texts)
            logger.info(f"Fitted tokenizer on batch {i//batch_size + 1}")
        
        vocab_size = tokenizer.get_vocab_size()
        logger.info(f"Final vocabulary size: {vocab_size:,}")
        
        # Update config
        TRAINING_CONFIG['vocab_size'] = vocab_size
        
        # Create training sequences
        logger.info("Creating training sequences...")
        X_train, y_train = create_training_sequences(texts, tokenizer, TRAINING_CONFIG['max_seq_len'])
        
        if len(X_train) == 0:
            raise ValueError("No training sequences created")
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Target data shape: {y_train.shape}")
        
        # Free up memory
        del texts
        
        # Create dataset with memory optimization
        logger.info("Creating optimized TensorFlow dataset...")
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = (train_dataset
                        .shuffle(buffer_size=1000)  # Smaller buffer for memory
                        .batch(TRAINING_CONFIG['batch_size'])
                        .prefetch(tf.data.AUTOTUNE)
                        .cache())  # Cache for better performance
        
        # Build large model
        logger.info("Building large MiniGPT model...")
        model = MiniGPT(vocab_size=vocab_size)
        
        # Compile with optimization for large model
        logger.info("Compiling model...")
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=TRAINING_CONFIG['learning_rate'],
            weight_decay=0.01,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=masked_sparse_categorical_crossentropy,
            metrics=[masked_accuracy],
            jit_compile=False  # Disable for better debugging
        )
        
        # Advanced callbacks for large model training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=2,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=1,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='minigpt_checkpoint_epoch_{epoch:02d}.weights.h5',
                monitor='loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                'training_log.csv',
                append=True
            )
        ]
        
        # Train model
        logger.info(f"Starting training for {TRAINING_CONFIG['num_epochs']} epochs...")
        logger.info(f"Batch size: {TRAINING_CONFIG['batch_size']}")
        logger.info(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
        
        history = model.fit(
            train_dataset,
            epochs=TRAINING_CONFIG['num_epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save everything
        logger.info("Saving model and tokenizer...")
        
        # Save final model weights
        model.save_weights('minigpt_final_large.weights.h5')
        logger.info("Model weights saved to 'minigpt_final_large.weights.h5'")
        
        # Save tokenizer
        with open('minigpt_tokenizer_large.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        logger.info("Tokenizer saved to 'minigpt_tokenizer_large.pkl'")
        
        # Save detailed configuration
        config_to_save = TRAINING_CONFIG.copy()
        config_to_save['training_samples'] = len(X_train)
        config_to_save['final_loss'] = float(history.history['loss'][-1])
        config_to_save['final_accuracy'] = float(history.history['masked_accuracy'][-1])
        config_to_save['training_history'] = {
            'loss': [float(x) for x in history.history['loss']],
            'masked_accuracy': [float(x) for x in history.history['masked_accuracy']]
        }
        config_to_save['timestamp'] = datetime.now().isoformat()
        config_to_save['gpu_used'] = gpu_available
        
        with open('minigpt_training_config_large.json', 'w') as f:
            json.dump(config_to_save, f, indent=2)
        logger.info("Training config saved to 'minigpt_training_config_large.json'")
        
        logger.info("=== LARGE MODEL TRAINING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Final loss: {history.history['loss'][-1]:.6f}")
        logger.info(f"Final accuracy: {history.history['masked_accuracy'][-1]:.6f}")
        
        return model, tokenizer, history
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Set TensorFlow logging level
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Run training
        model, tokenizer, history = train_model()
        
        print("\n" + "="*60)
        print("LARGE MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Files saved:")
        print("- minigpt_final_large.weights.h5 (final model weights)")
        print("- minigpt_tokenizer_large.pkl (tokenizer)")
        print("- minigpt_training_config_large.json (detailed configuration)")
        print("- minigpt_checkpoint_epoch_*.weights.h5 (epoch checkpoints)")
        print("- training_log.csv (training metrics log)")
        
    except Exception as e:
        print(f"\nTRAINING FAILED: {e}")
        logger.error(f"Main execution failed: {e}")
        exit(1)