import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pickle
import os
from datetime import datetime
import json
import re
import datasets
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set TensorFlow configuration
tf.config.threading.set_inter_op_parallelism_threads(12)
tf.config.threading.set_intra_op_parallelism_threads(12)
# Remove deprecated JIT setting
# tf.config.optimizer.set_jit(True)  # This is deprecated

CONFIG = {
    'vocab_size': 20000,
    'seq_len': 256,
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.0001,
    'dropout_rate': 0.1,
    'temperature': 1.2,
    'max_response_length': 60,
    'target_tokens': 500000
}

class SimpleTokenizer:
    """Simple tokenizer to replace the missing ChatTokenizer"""
    def __init__(self, vocab_size=20000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.current_id = 4
        
    def fit_on_texts(self, texts):
        """Build vocabulary from texts"""
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words[:self.vocab_size - len(self.word_to_id)]:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.current_id
                self.id_to_word[self.current_id] = word
                self.current_id += 1
    
    def texts_to_sequences(self, texts):
        """Convert texts to sequences of token IDs"""
        sequences = []
        for text in texts:
            words = text.lower().split()
            sequence = [self.word_to_id.get(word, 1) for word in words]  # 1 is <UNK>
            sequences.append(sequence)
        return sequences
    
    def sequences_to_texts(self, sequences):
        """Convert sequences back to texts"""
        texts = []
        for sequence in sequences:
            words = [self.id_to_word.get(id, '<UNK>') for id in sequence if id != 0]
            texts.append(' '.join(words))
        return texts
    
    def get_vocab_size(self):
        return len(self.word_to_id)

class SimpleTransformer(tf.keras.Model):
    """Simple transformer model to replace MiniGPT"""
    def __init__(self, vocab_size, max_seq_len=256, embed_dim=512, num_heads=8, num_layers=6, ffn_dim=1024, dropout_rate=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Embedding layers
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embedding = tf.keras.layers.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = []
        for _ in range(num_layers):
            block = tf.keras.Sequential([
                tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(ffn_dim, activation='relu'),
                tf.keras.layers.Dense(embed_dim),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.LayerNormalization()
            ])
            self.transformer_blocks.append(block)
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(vocab_size)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        seq_len = tf.shape(inputs)[1]
        
        # Token embeddings
        token_emb = self.token_embedding(inputs)
        
        # Position embeddings
        positions = tf.range(seq_len)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_emb + pos_emb
        x = self.dropout(x, training=training)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            # Multi-head attention
            attn_output = block.layers[0](x, x, training=training)
            attn_output = block.layers[1](attn_output, training=training)
            x = block.layers[2](x + attn_output)
            
            # Feed-forward network
            ffn_output = block.layers[3](x)
            ffn_output = block.layers[4](ffn_output)
            ffn_output = block.layers[5](ffn_output, training=training)
            x = block.layers[6](x + ffn_output)
        
        # Output projection
        return self.output_layer(x)

def download_persona_chat():
    """Download PersonaChat dataset with better error handling"""
    try:
        dataset = datasets.load_dataset("conv_ai_2", split="train")
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if (isinstance(dialog[i], dict) and 'text' in dialog[i] and 
                        isinstance(dialog[i+1], dict) and 'text' in dialog[i+1]):
                        q_text = dialog[i]['text'].strip()
                        a_text = dialog[i+1]['text'].strip()
                        if q_text and a_text:
                            texts.append(f"Q: {q_text}")
                            texts.append(f"A: {a_text}")
        logger.info(f"PersonaChat: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load PersonaChat: {e}")
        return []

def download_daily_dialog():
    """Download DailyDialog dataset with better error handling"""
    try:
        dataset = datasets.load_dataset("daily_dialog", split="train")
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    q_text = str(dialog[i]).strip()
                    a_text = str(dialog[i+1]).strip()
                    if q_text and a_text and len(q_text) > 2 and len(a_text) > 2:
                        texts.append(f"Q: {q_text}")
                        texts.append(f"A: {a_text}")
        logger.info(f"DailyDialog: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load DailyDialog: {e}")
        return []

def create_fallback_conversation_data():
    """Create fallback conversation data when datasets fail to load"""
    conversation_patterns = [
        ("Q: Hello", "A: Hi there! How can I help you today?"),
        ("Q: Hi there", "A: Hello! Nice to meet you. What's on your mind?"),
        ("Q: How are you?", "A: I'm doing well, thank you for asking! How about you?"),
        ("Q: What's your name?", "A: I'm an AI assistant. You can just call me Assistant."),
        ("Q: Can you help me?", "A: Absolutely! I'm here to help. What do you need assistance with?"),
        ("Q: Thank you", "A: You're very welcome! Is there anything else I can help you with?"),
        ("Q: Tell me a joke", "A: Why don't scientists trust atoms? Because they make up everything!"),
        ("Q: What's the weather like?", "A: I don't have access to current weather data, but you can check a weather app."),
        ("Q: What time is it?", "A: I don't have access to the current time. Please check your device's clock."),
        ("Q: Goodbye", "A: Goodbye! It was wonderful chatting with you. Take care!"),
    ]
    
    texts = []
    # Repeat patterns multiple times to create more training data
    for _ in range(1000):
        for q, a in conversation_patterns:
            texts.append(q)
            texts.append(a)
    
    logger.info(f"Fallback conversation data: {len(texts)} texts created")
    return texts

def prepare_conversation_data():
    """Load and prepare conversation datasets with fallback"""
    logger.info("Loading conversation datasets...")
    all_texts = []
    
    # Try to load real datasets
    dataset_loaders = [
        ('PersonaChat', download_persona_chat),
        ('DailyDialog', download_daily_dialog),
    ]
    
    for name, loader_func in dataset_loaders:
        try:
            logger.info(f"Loading {name}...")
            texts = loader_func()
            all_texts.extend(texts)
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            continue
    
    # If no datasets loaded successfully, use fallback
    if len(all_texts) == 0:
        logger.warning("No external datasets loaded, using fallback conversation data")
        all_texts = create_fallback_conversation_data()
    
    # Add some basic conversation patterns regardless
    fallback_texts = create_fallback_conversation_data()
    all_texts.extend(fallback_texts)
    
    logger.info(f"Total training texts: {len(all_texts):,}")
    logger.info(f"Estimated Q&A pairs: {len(all_texts) // 2:,}")
    
    return all_texts

def create_training_data(texts, tokenizer, seq_len):
    """Create training sequences with proper error handling"""
    logger.info("Creating training sequences...")
    
    inputs, targets = [], []
    pad_token_id = 0  # <PAD> token
    
    for i in range(0, len(texts)-1, 2):
        if i+1 < len(texts) and texts[i].startswith('Q:') and texts[i+1].startswith('A:'):
            # Combine question and answer
            combined_text = f"{texts[i][2:].strip()} {texts[i+1][2:].strip()}"
            
            try:
                sequence = tokenizer.texts_to_sequences([combined_text])[0]
                
                if len(sequence) > 1:  # Ensure we have at least 2 tokens
                    # Pad or truncate to seq_len + 1
                    if len(sequence) <= seq_len:
                        padded = sequence + [pad_token_id] * (seq_len + 1 - len(sequence))
                    else:
                        padded = sequence[:seq_len + 1]
                    
                    # Create input-target pairs
                    inputs.append(padded[:-1])
                    targets.append(padded[1:])
                    
            except Exception as e:
                logger.warning(f"Error processing sequence: {e}")
                continue
    
    logger.info(f"Created {len(inputs):,} training sequences")
    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """Masked loss function for padded sequences"""
    # Create mask for non-padding tokens
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    
    # Calculate sparse categorical crossentropy
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    
    # Apply mask
    masked_loss = loss * mask
    
    # Return average loss over non-padded tokens
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    """Masked accuracy for padded sequences"""
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    
    correct = tf.cast(tf.equal(y_true, predictions), tf.float32) * mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)

def train_conversation_model():
    """Main training function with proper error handling"""
    logger.info("=== CONVERSATION CHATBOT TRAINING ===")
    
    try:
        # Load and prepare data
        texts = prepare_conversation_data()
        
        # Create tokenizer
        logger.info("Creating tokenizer...")
        tokenizer = SimpleTokenizer(vocab_size=CONFIG['vocab_size'])
        tokenizer.fit_on_texts(texts)
        
        logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
        
        # Create training data
        X_train, y_train = create_training_data(texts, tokenizer, CONFIG['seq_len'])
        
        if len(X_train) == 0:
            raise ValueError("No training data created")
        
        logger.info(f"Training data shape: {X_train.shape}")
        
        # Create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = (train_dataset
                        .shuffle(buffer_size=min(10000, len(X_train)))
                        .batch(CONFIG['batch_size'])
                        .prefetch(tf.data.AUTOTUNE))
        
        # Build model
        logger.info("Building model...")
        model = SimpleTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            max_seq_len=CONFIG['seq_len'],
            embed_dim=256,  # Reduced for stability
            num_heads=8,
            num_layers=4,   # Reduced for stability
            ffn_dim=512,    # Reduced for stability
            dropout_rate=CONFIG['dropout_rate']
        )
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss=masked_sparse_categorical_crossentropy,
            metrics=[masked_accuracy]
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'chatbot_model_weights.h5',
                monitor='loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Starting training...")
        history = model.fit(
            train_dataset,
            epochs=CONFIG['num_epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and tokenizer
        logger.info("Saving model and tokenizer...")
        model.save_weights('chatbot_final_weights.h5')
        
        with open('chatbot_tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        
        # Save config
        config_to_save = CONFIG.copy()
        config_to_save['vocab_size_actual'] = tokenizer.get_vocab_size()
        config_to_save['training_samples'] = len(X_train)
        config_to_save['timestamp'] = datetime.now().isoformat()
        
        with open('chatbot_config.json', 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        logger.info("Training completed successfully!")
        return model, tokenizer, history
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def generate_response(model, tokenizer, input_text, max_length=50):
    """Generate response from trained model"""
    try:
        # Tokenize input
        input_sequence = tokenizer.texts_to_sequences([f"Q: {input_text}"])[0]
        
        # Pad sequence
        if len(input_sequence) < CONFIG['seq_len']:
            input_sequence = input_sequence + [0] * (CONFIG['seq_len'] - len(input_sequence))
        else:
            input_sequence = input_sequence[:CONFIG['seq_len']]
        
        # Convert to tensor
        input_tensor = tf.expand_dims(input_sequence, 0)
        
        # Generate response
        output = model(input_tensor, training=False)
        predicted_ids = tf.argmax(output, axis=-1)[0]
        
        # Convert back to text
        response_text = tokenizer.sequences_to_texts([predicted_ids.numpy()])[0]
        return response_text
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return "I'm sorry, I couldn't generate a response."

if __name__ == "__main__":
    try:
        model, tokenizer, history = train_conversation_model()
        
        # Test the model
        print("\n=== Testing the model ===")
        test_inputs = ["Hello", "How are you?", "What's your name?", "Can you help me?"]
        
        for test_input in test_inputs:
            response = generate_response(model, tokenizer, test_input)
            print(f"Input: {test_input}")
            print(f"Response: {response}")
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"Training failed with error: {e}")