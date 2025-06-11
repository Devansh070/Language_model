import tensorflow as tf
import numpy as np
import pickle
import os
from datetime import datetime
import json
import logging
import datasets
from datasets import disable_caching, load_dataset
from minigpt_transformer import EnhancedMiniGPT, ModelConfig
from transformers import AutoTokenizer
import time
from tqdm import tqdm
import psutil
import gc

# Disable datasets caching to prevent disk space issues
disable_caching()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

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

# Training configuration aligned with ModelConfig
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 5,
    'learning_rate': 1e-4,
    'max_seq_len': 512,
    'target_tokens': 10000000,
    'max_samples_per_dataset': 100000,
    'validation_split': 0.1,  # 10% validation split
}

def safe_dataset_load(dataset_name, split='train', max_samples=10000):
    """Safely load a dataset with error handling and progress tracking"""
    try:
        logger.info(f"Loading dataset: {dataset_name}")
        start_time = time.time()
        
        # Load dataset with progress tracking
        dataset = load_dataset(dataset_name, split=split, streaming=False)
        
        # Take subset if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        load_time = time.time() - start_time
        logger.info(f"Successfully loaded {len(dataset)} samples from {dataset_name} in {load_time:.2f} seconds")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
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
                                texts.append(f"Q: {q_text}\nA: {a_text}")
                
                elif 'conversations' in example:
                    # Handle conversation datasets like ShareGPT
                    convs = example['conversations']
                    if isinstance(convs, list) and len(convs) >= 2:
                        for i in range(0, len(convs)-1, 2):
                            if i+1 < len(convs):
                                if isinstance(convs[i], dict) and isinstance(convs[i+1], dict):
                                    q_text = convs[i].get('value', '').strip()
                                    a_text = convs[i+1].get('value', '').strip()
                                else:
                                    q_text = str(convs[i]).strip()
                                    a_text = str(convs[i+1]).strip()
                                if q_text and a_text:
                                    texts.append(f"Q: {q_text}\nA: {a_text}")
                
                elif 'utterance' in example and 'context' in example:
                    # Handle empathetic dialogues
                    context = example['context'].strip()
                    utterance = example['utterance'].strip()
                    if context and utterance:
                        texts.append(f"Q: {context}\nA: {utterance}")
                
                elif 'prompt' in example and 'response' in example:
                    # Handle prompt-response datasets
                    prompt = example['prompt'].strip()
                    response = example['response'].strip()
                    if prompt and response:
                        texts.append(f"Q: {prompt}\nA: {response}")
                
                elif 'input' in example and 'output' in example:
                    # Handle input-output datasets
                    inp = example['input'].strip()
                    out = example['output'].strip()
                    if inp and out:
                        texts.append(f"Q: {inp}\nA: {out}")
                
                elif 'text' in example and 'summary' in example:
                    # Handle text-summary datasets
                    text = example['text'].strip()
                    summary = example['summary'].strip()
                    if text and summary:
                        texts.append(f"Q: Summarize this: {text}\nA: {summary}")
                
                elif 'question' in example and 'answer' in example:
                    # Handle Q&A datasets
                    question = example['question'].strip()
                    answer = example['answer'].strip()
                    if question and answer:
                        texts.append(f"Q: {question}\nA: {answer}")
                
                elif 'instruction' in example:
                    # Handle Alpaca-style datasets
                    instruction = example['instruction'].strip()
                    output = example.get('output', '').strip()
                    inp = example.get('input', '').strip()
                    
                    if instruction and output:
                        if inp:
                            texts.append(f"Q: {instruction}\nInput: {inp}\nA: {output}")
                        else:
                            texts.append(f"Q: {instruction}\nA: {output}")
                
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

def load_all_conversation_datasets(max_samples=10000):
    """Load all conversation datasets with progress tracking"""
    datasets_to_try = [
        'daily_dialog',
        'empathetic_dialogues',
        'tatsu-lab/alpaca',
    ]
    
    all_texts = []
    total_samples = 0
    
    for dataset_name in datasets_to_try:
        dataset = safe_dataset_load(dataset_name, max_samples=max_samples)
        if dataset is not None:
            texts = extract_conversations_from_dataset(dataset, dataset_name)
            all_texts.extend(texts)
            total_samples += len(texts)
            logger.info(f"Added {len(texts)} samples from {dataset_name}")
    
    # If no datasets loaded, create some simple training data
    if len(all_texts) == 0:
        logger.warning("No datasets loaded, creating simple training data")
        all_texts = [
            "Q: What is the capital of France?\nA: The capital of France is Paris.",
            "Q: How are you?\nA: I'm doing well, thank you for asking!",
            "Q: What is machine learning?\nA: Machine learning is a branch of artificial intelligence.",
            "Q: Tell me a joke.\nA: Why don't scientists trust atoms? Because they make up everything!",
            "Q: What's 2+2?\nA: 2+2 equals 4.",
        ] * 1000  # Repeat for more training data
    
    logger.info(f"Total samples collected: {len(all_texts)}")
    return all_texts

def create_sequences(texts, tokenizer, max_length=512, batch_size=1000):
    """Create training sequences using the tokenizer"""
    logger.info("Creating training sequences...")
    start_time = time.time()
    
    all_sequences = []
    total_tokens = 0
    
    # Process in batches to manage memory
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        encoded = tokenizer(
            batch_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        # Add to sequences
        all_sequences.extend(encoded['input_ids'])
        total_tokens += tf.reduce_sum(tf.cast(encoded['attention_mask'], tf.int32))
        
        # Log memory usage periodically
        if i % (batch_size * 10) == 0:
            log_memory_usage()
            gc.collect()
    
    logger.info(f"Created {len(all_sequences)} sequences with {total_tokens} tokens in {time.time() - start_time:.2f} seconds")
    return tf.data.Dataset.from_tensor_slices(all_sequences)

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """Custom loss function that ignores padding tokens"""
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    """Custom accuracy metric that ignores padding tokens"""
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    pred = tf.argmax(y_pred, axis=-1)
    correct = tf.cast(tf.equal(y_true, pred), tf.float32)
    return tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)

def train_model():
    """Main training function"""
    try:
        # Setup GPU
        has_gpu = setup_gpu()
        
        # Load and initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load datasets
        logger.info("Loading conversation datasets...")
        texts = load_all_conversation_datasets(max_samples=TRAINING_CONFIG['max_samples_per_dataset'])
        
        if len(texts) == 0:
            raise ValueError("No texts loaded from datasets")
        
        # Create training sequences
        logger.info("Creating training sequences...")
        sequences = create_sequences(texts, tokenizer)
        
        if len(sequences) == 0:
            raise ValueError("No training sequences created")
        
        logger.info(f"Training data shape: {sequences.element_spec}")
        
        # Free up memory
        del texts
        gc.collect()
        
        # Split into train and validation
        val_size = int(len(sequences) * TRAINING_CONFIG['validation_split'])
        train_dataset = sequences.skip(val_size).shuffle(buffer_size=1000)
        val_dataset = sequences.take(val_size)
        
        # Create optimized datasets
        train_dataset = train_dataset.batch(TRAINING_CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(TRAINING_CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
        
        # Create model configuration
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=TRAINING_CONFIG['max_seq_len'],
            embed_dim=512,
            num_heads=8,
            num_layers=12,
            ffn_dim=2048,
            dropout=0.1,
            layer_norm_epsilon=1e-5,
            use_custom_attention=True  # Enable Custom Multi-Head Attention
        )
        
        # Build model
        logger.info("Building MiniGPT model...")
        model = EnhancedMiniGPT(config=config)
        
        # Compile model with custom metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=TRAINING_CONFIG['learning_rate']),
            loss=masked_sparse_categorical_crossentropy,
            metrics=[masked_accuracy]
        )
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'minigpt_checkpoint_{epoch}',
                save_weights_only=True,
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        # Train model with validation
        logger.info("Starting training...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=TRAINING_CONFIG['num_epochs'],
            callbacks=callbacks
        )
        
        # Save final model
        model.save_weights('minigpt_final')
        
        # Save training history
        with open('training_history.json', 'w') as f:
            json.dump(history.history, f)
        
        logger.info("Training completed successfully!")
        return model, history
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()