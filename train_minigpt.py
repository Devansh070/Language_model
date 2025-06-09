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
    'target_tokens': 10000000  # Updated to 10M tokens
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
        # Create a specific cache directory
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Try different dataset names with offline mode
        try:
            dataset = datasets.load_dataset("conv_ai_2", split="train", cache_dir=cache_dir, local_files_only=True)
        except:
            try:
                dataset = datasets.load_dataset("conv_ai", "conv_ai_2", split="train", cache_dir=cache_dir, local_files_only=True)
            except:
                # If offline loading fails, try online loading
                dataset = datasets.load_dataset("conv_ai_2", split="train", cache_dir=cache_dir)
            
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
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            dataset = datasets.load_dataset("daily_dialog", split="train", cache_dir=cache_dir, local_files_only=True)
        except:
            # If offline loading fails, try online loading
            dataset = datasets.load_dataset("daily_dialog", split="train", cache_dir=cache_dir)
        
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

def download_reddit_conversations():
    """Download Reddit conversations dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Try different Reddit conversation datasets with offline mode
        try:
            dataset = datasets.load_dataset("reddit", "conversations", split="train", cache_dir=cache_dir, local_files_only=True)
        except:
            try:
                dataset = datasets.load_dataset("reddit_tifu", "short", split="train", cache_dir=cache_dir, local_files_only=True)
            except:
                # If offline loading fails, try online loading
                dataset = datasets.load_dataset("reddit", "conversations", split="train", cache_dir=cache_dir)
            
        texts = []
        for example in dataset:
            if 'conversation' in example:
                conv = example['conversation']
                for i in range(len(conv) - 1):
                    if conv[i].strip() and conv[i+1].strip():
                        texts.append(f"Q: {conv[i].strip()}")
                        texts.append(f"A: {conv[i+1].strip()}")
            elif 'tldr' in example and 'text' in example:
                texts.append(f"Q: {example['text'].strip()}")
                texts.append(f"A: {example['tldr'].strip()}")
        logger.info(f"Reddit conversations: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load Reddit conversations: {e}")
        return []

def download_empathetic_dialogues():
    """Download EmpatheticDialogues dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("empathetic_dialogues", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'utterance' in example and 'context' in example:
                context = example['context'].strip()
                utterance = example['utterance'].strip()
                if context and utterance:
                    texts.append(f"Q: {context}")
                    texts.append(f"A: {utterance}")
        logger.info(f"EmpatheticDialogues: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load EmpatheticDialogues: {e}")
        return []

def download_blended_skill_talk():
    """Download BlendedSkillTalk dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("blended_skill_talk", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"BlendedSkillTalk: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load BlendedSkillTalk: {e}")
        return []

def download_wizard_of_wikipedia():
    """Download Wizard of Wikipedia dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("wizard_of_wikipedia", "generated", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"Wizard of Wikipedia: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load Wizard of Wikipedia: {e}")
        return []

def download_conv_ai_3():
    """Download ConvAI3 dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("conv_ai_3", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"ConvAI3: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ConvAI3: {e}")
        return []

def download_conv_ai_2():
    """Download ConvAI2 dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("conv_ai_2", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"ConvAI2: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ConvAI2: {e}")
        return []

def download_conv_ai():
    """Download ConvAI dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("conv_ai", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"ConvAI: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ConvAI: {e}")
        return []

def download_conv_ai_2_personachat():
    """Download ConvAI2 PersonaChat dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("conv_ai_2", "personachat", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"ConvAI2 PersonaChat: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ConvAI2 PersonaChat: {e}")
        return []

def download_conv_ai_2_convai2():
    """Download ConvAI2 dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("conv_ai_2", "convai2", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"ConvAI2: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ConvAI2: {e}")
        return []

def download_conv_ai_2_convai2_inferred():
    """Download ConvAI2 Inferred dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("conv_ai_2", "convai2_inferred", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"ConvAI2 Inferred: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ConvAI2 Inferred: {e}")
        return []

def download_conv_ai_2_convai2_inferred_original():
    """Download ConvAI2 Inferred Original dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("conv_ai_2", "convai2_inferred_original", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"ConvAI2 Inferred Original: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ConvAI2 Inferred Original: {e}")
        return []

def download_conv_ai_2_convai2_inferred_original_personachat():
    """Download ConvAI2 Inferred Original PersonaChat dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("conv_ai_2", "convai2_inferred_original_personachat", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"ConvAI2 Inferred Original PersonaChat: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ConvAI2 Inferred Original PersonaChat: {e}")
        return []

def download_conv_ai_2_convai2_inferred_original_convai2():
    """Download ConvAI2 Inferred Original ConvAI2 dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("conv_ai_2", "convai2_inferred_original_convai2", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"ConvAI2 Inferred Original ConvAI2: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ConvAI2 Inferred Original ConvAI2: {e}")
        return []

def download_conv_ai_2_convai2_inferred_original_convai2_inferred():
    """Download ConvAI2 Inferred Original ConvAI2 Inferred dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("conv_ai_2", "convai2_inferred_original_convai2_inferred", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"ConvAI2 Inferred Original ConvAI2 Inferred: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ConvAI2 Inferred Original ConvAI2 Inferred: {e}")
        return []

def download_conv_ai_2_convai2_inferred_original_convai2_inferred_original():
    """Download ConvAI2 Inferred Original ConvAI2 Inferred Original dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        dataset = datasets.load_dataset("conv_ai_2", "convai2_inferred_original_convai2_inferred_original", split="train", cache_dir=cache_dir)
        texts = []
        for example in dataset:
            if 'dialog' in example:
                dialog = example['dialog']
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        texts.append(f"Q: {dialog[i].strip()}")
                        texts.append(f"A: {dialog[i+1].strip()}")
        logger.info(f"ConvAI2 Inferred Original ConvAI2 Inferred Original: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ConvAI2 Inferred Original ConvAI2 Inferred Original: {e}")
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

def download_alpaca():
    """Download Alpaca dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            dataset = datasets.load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir, local_files_only=True)
        except:
            # If offline loading fails, try online loading
            dataset = datasets.load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir)
        
        texts = []
        for example in dataset:
            if 'instruction' in example and 'output' in example:
                instruction = example['instruction'].strip()
                output = example['output'].strip()
                if instruction and output:
                    texts.append(f"Q: {instruction}")
                    texts.append(f"A: {output}")
        logger.info(f"Alpaca: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load Alpaca: {e}")
        return []

def download_dolly():
    """Download Dolly dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            dataset = datasets.load_dataset("databricks/databricks-dolly-15k", split="train", cache_dir=cache_dir, local_files_only=True)
        except:
            # If offline loading fails, try online loading
            dataset = datasets.load_dataset("databricks/databricks-dolly-15k", split="train", cache_dir=cache_dir)
        
        texts = []
        for example in dataset:
            if 'instruction' in example and 'response' in example:
                instruction = example['instruction'].strip()
                response = example['response'].strip()
                if instruction and response:
                    texts.append(f"Q: {instruction}")
                    texts.append(f"A: {response}")
        logger.info(f"Dolly: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load Dolly: {e}")
        return []

def download_anthropic_hh():
    """Download Anthropic HH dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            dataset = datasets.load_dataset("Anthropic/hh-rlhf", split="train", cache_dir=cache_dir, local_files_only=True)
        except:
            # If offline loading fails, try online loading
            dataset = datasets.load_dataset("Anthropic/hh-rlhf", split="train", cache_dir=cache_dir)
        
        texts = []
        for example in dataset:
            if 'chosen' in example:
                chosen = example['chosen'].strip()
                if chosen:
                    # Split the chosen text into turns
                    turns = chosen.split('\n\n')
                    for i in range(len(turns)-1):
                        if turns[i].strip() and turns[i+1].strip():
                            texts.append(f"Q: {turns[i].strip()}")
                            texts.append(f"A: {turns[i+1].strip()}")
        logger.info(f"Anthropic HH: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load Anthropic HH: {e}")
        return []

def download_sharegpt():
    """Download ShareGPT dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            dataset = datasets.load_dataset("AlekseyKorshuk/sharegpt", split="train", cache_dir=cache_dir, local_files_only=True)
        except:
            # If offline loading fails, try online loading
            dataset = datasets.load_dataset("AlekseyKorshuk/sharegpt", split="train", cache_dir=cache_dir)
        
        texts = []
        for example in dataset:
            if 'conversations' in example:
                conv = example['conversations']
                for i in range(len(conv)-1):
                    if conv[i].strip() and conv[i+1].strip():
                        texts.append(f"Q: {conv[i].strip()}")
                        texts.append(f"A: {conv[i+1].strip()}")
        logger.info(f"ShareGPT: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load ShareGPT: {e}")
        return []

def download_lima():
    """Download LIMA dataset"""
    try:
        cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            dataset = datasets.load_dataset("GAIR/lima", split="train", cache_dir=cache_dir, local_files_only=True)
        except:
            # If offline loading fails, try online loading
            dataset = datasets.load_dataset("GAIR/lima", split="train", cache_dir=cache_dir)
        
        texts = []
        for example in dataset:
            if 'conversations' in example:
                conv = example['conversations']
                for i in range(len(conv)-1):
                    if conv[i].strip() and conv[i+1].strip():
                        texts.append(f"Q: {conv[i].strip()}")
                        texts.append(f"A: {conv[i+1].strip()}")
        logger.info(f"LIMA: {len(texts)} texts loaded")
        return texts
    except Exception as e:
        logger.error(f"Failed to load LIMA: {e}")
        return []

def prepare_conversation_data():
    """Load and prepare conversation datasets with fallback"""
    logger.info("Loading conversation datasets...")
    all_texts = []
    
    # Try to load real datasets
    dataset_loaders = [
        ('PersonaChat', download_persona_chat),
        ('DailyDialog', download_daily_dialog),
        ('Reddit', download_reddit_conversations),
        ('EmpatheticDialogues', download_empathetic_dialogues),
        ('BlendedSkillTalk', download_blended_skill_talk),
        ('WizardOfWikipedia', download_wizard_of_wikipedia),
        ('ConvAI3', download_conv_ai_3),
        ('ConvAI2', download_conv_ai_2),
        ('ConvAI', download_conv_ai),
        ('ConvAI2 PersonaChat', download_conv_ai_2_personachat),
        ('ConvAI2 ConvAI2', download_conv_ai_2_convai2),
        ('ConvAI2 Inferred', download_conv_ai_2_convai2_inferred),
        ('ConvAI2 Inferred Original', download_conv_ai_2_convai2_inferred_original),
        ('ConvAI2 Inferred Original PersonaChat', download_conv_ai_2_convai2_inferred_original_personachat),
        ('ConvAI2 Inferred Original ConvAI2', download_conv_ai_2_convai2_inferred_original_convai2),
        ('ConvAI2 Inferred Original ConvAI2 Inferred', download_conv_ai_2_convai2_inferred_original_convai2_inferred),
        ('ConvAI2 Inferred Original ConvAI2 Inferred Original', download_conv_ai_2_convai2_inferred_original_convai2_inferred_original),
        ('Alpaca', download_alpaca),
        ('Dolly', download_dolly),
        ('Anthropic HH', download_anthropic_hh),
        ('ShareGPT', download_sharegpt),
        ('LIMA', download_lima),
    ]
    
    for name, loader_func in dataset_loaders:
        try:
            logger.info(f"Loading {name}...")
            texts = loader_func()
            if texts:  # Only extend if we got some texts
                all_texts.extend(texts)
                total_tokens = sum(len(text.split()) for text in all_texts)
                logger.info(f"Total texts so far: {len(all_texts):,}")
                logger.info(f"Total tokens so far: {total_tokens:,}")
                
                # Check if we've reached our target
                if total_tokens >= CONFIG['target_tokens']:
                    logger.info(f"Reached target of {CONFIG['target_tokens']:,} tokens")
                    break
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            continue
    
    # If we haven't reached our target, use fallback data
    if len(all_texts) == 0:
        logger.warning("No external datasets loaded, using fallback conversation data")
        all_texts = create_fallback_conversation_data()
    
    # Add some basic conversation patterns regardless
    fallback_texts = create_fallback_conversation_data()
    all_texts.extend(fallback_texts)
    
    # Calculate total tokens
    total_tokens = sum(len(text.split()) for text in all_texts)
    
    logger.info(f"Total training texts: {len(all_texts):,}")
    logger.info(f"Estimated Q&A pairs: {len(all_texts) // 2:,}")
    logger.info(f"Total tokens (without padding): {total_tokens:,}")
    
    return all_texts

def create_training_data(texts, tokenizer, seq_len):
    """Create training sequences with proper error handling"""
    logger.info("Creating training sequences...")
    
    inputs, targets = [], []
    pad_token_id = 0  # <PAD> token
    total_tokens = 0
    
    for i in range(0, len(texts)-1, 2):
        if i+1 < len(texts) and texts[i].startswith('Q:') and texts[i+1].startswith('A:'):
            # Combine question and answer
            combined_text = f"{texts[i][2:].strip()} {texts[i+1][2:].strip()}"
            
            try:
                sequence = tokenizer.texts_to_sequences([combined_text])[0]
                
                if len(sequence) > 1:  # Ensure we have at least 2 tokens
                    total_tokens += len(sequence)
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
    logger.info(f"Total non-padding tokens: {total_tokens:,}")
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

def masked_perplexity(y_true, y_pred):
    """Calculate perplexity for masked sequences"""
    # Create mask for non-padding tokens
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    
    # Calculate cross entropy loss
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    
    # Apply mask
    masked_loss = loss * mask
    
    # Calculate average loss over non-padded tokens
    avg_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
    
    # Calculate perplexity
    perplexity = tf.exp(avg_loss)
    return perplexity

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
            metrics=[masked_accuracy, masked_perplexity]  # Added perplexity metric
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
                'chatbot_model.weights.h5',  # Fixed filepath with correct extension
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
        model.save_weights('chatbot_final.weights.h5')  # Fixed filepath with correct extension
        
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
        
        print("\n=== Chatbot is ready! Type 'quit' to exit ===")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
                
            response = generate_response(model, tokenizer, user_input)
            print(f"Bot: {response}")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"Training failed with error: {e}")