import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pickle
import os
from minigpt_transformer import MiniGPT, build_chat_model, ChatTokenizer
import math
import json
from datetime import datetime
import datasets
from transformers import AutoTokenizer
import requests
import gzip
import re

tf.config.threading.set_inter_op_parallelism_threads(12)
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.optimizer.set_jit(True)

CONFIG = {
    'vocab_size': 15000,  
    'seq_len': 256,       
    'batch_size': 32,     
    'num_epochs': 10,     
    'learning_rate': 0.0002,
    'dropout_rate': 0.1,
    'temperature': 1.0,
    'max_response_length': 80,
    'target_tokens': 500_000,  
    'warmup_steps': 1000,
    'gradient_accumulation_steps': 4
}

def download_persona_chat():
    """Download PersonaChat dataset"""
    try:
        dataset = datasets.load_dataset("conv_ai_2", split="train")
        texts = []
        for example in dataset:
            dialog = example['dialog']
            for i in range(len(dialog) - 1):
                if dialog[i]['text'].strip() and dialog[i+1]['text'].strip():
                    texts.append(f"Q: {dialog[i]['text'].strip()}")
                    texts.append(f"A: {dialog[i+1]['text'].strip()}")
        print(f"PersonaChat: {len(texts)} texts")
        return texts
    except Exception as e:
        print(f"Failed to load PersonaChat: {e}")
        return []

def download_daily_dialog():
    """Download DailyDialog dataset"""
    try:
        dataset = datasets.load_dataset("daily_dialog", split="train")
        texts = []
        for example in dataset:
            dialog = example['dialog']
            for i in range(len(dialog) - 1):
                if dialog[i].strip() and dialog[i+1].strip():
                    texts.append(f"Q: {dialog[i].strip()}")
                    texts.append(f"A: {dialog[i+1].strip()}")
        print(f"DailyDialog: {len(texts)} texts")
        return texts
    except Exception as e:
        print(f"Failed to load DailyDialog: {e}")
        return []

def download_blended_skill_talk():
    """Download BlendedSkillTalk dataset"""
    try:
        dataset = datasets.load_dataset("blended_skill_talk", split="train")
        texts = []
        for example in dataset:
            dialog = example['dialog']
            for i in range(len(dialog) - 1):
                if dialog[i]['text'].strip() and dialog[i+1]['text'].strip():
                    texts.append(f"Q: {dialog[i]['text'].strip()}")
                    texts.append(f"A: {dialog[i+1]['text'].strip()}")
        print(f"BlendedSkillTalk: {len(texts)} texts")
        return texts
    except Exception as e:
        print(f"Failed to load BlendedSkillTalk: {e}")
        return []

def download_opensubtitles():
    """Download OpenSubtitles dialog data"""
    try:
        dataset = datasets.load_dataset("open_subtitles", "en", split="train", streaming=True)
        texts = []
        count = 0
        target_pairs = 500000  
        for example in dataset:
            if count >= target_pairs:
                break
            
            sentences = example['translation']['en']
            if len(sentences) >= 2:
                for i in range(len(sentences) - 1):
                    if sentences[i].strip() and sentences[i+1].strip():
                        q = re.sub(r'[<>{}[\]]', '', sentences[i]).strip()
                        a = re.sub(r'[<>{}[\]]', '', sentences[i+1]).strip()
                        if len(q) > 5 and len(a) > 5 and len(q) < 200 and len(a) < 200:
                            texts.append(f"Q: {q}")
                            texts.append(f"A: {a}")
                            count += 1
        
        print(f"OpenSubtitles: {len(texts)} texts")
        return texts
    except Exception as e:
        print(f"Failed to load OpenSubtitles: {e}")
        return []

def download_wizard_of_wikipedia():
    """Download Wizard of Wikipedia dataset"""
    try:
        dataset = datasets.load_dataset("wizard_of_wikipedia", split="train")
        texts = []
        for example in dataset:
            dialog = example['dialog']
            for i in range(len(dialog) - 1):
                if dialog[i]['text'].strip() and dialog[i+1]['text'].strip():
                    texts.append(f"Q: {dialog[i]['text'].strip()}")
                    texts.append(f"A: {dialog[i+1]['text'].strip()}")
        print(f"Wizard of Wikipedia: {len(texts)} texts")
        return texts
    except Exception as e:
        print(f"Failed to load Wizard of Wikipedia: {e}")
        return []

def download_reddit_conversations():
    """Simulate Reddit conversation data (replace with actual Reddit API)"""
    reddit_patterns = [
        ("Q: What's the best way to learn programming?", "A: Start with Python, it's beginner-friendly and has great documentation."),
        ("Q: Anyone else having trouble sleeping lately?", "A: Yes! I've been trying meditation before bed and it helps a bit."),
        ("Q: What's your favorite book recommendation?", "A: Depends on the genre, but for sci-fi I'd say Dune is amazing."),
        ("Q: How do you stay motivated?", "A: I break big goals into smaller tasks and celebrate small wins."),
        ("Q: Best coffee brewing method?", "A: Pour-over gives you the most control, but French press is easier for daily use."),
    ]
    
    texts = []
    for _ in range(1000):  
        for q, a in reddit_patterns:
            texts.append(q)
            texts.append(a)
    
    print(f"Reddit-style conversations: {len(texts)} texts")
    return texts

def prepare_comprehensive_conversation_data():
    """Load and prepare comprehensive conversation datasets"""
    print("Loading comprehensive conversation datasets...")
    all_texts = []
    
    dataset_loaders = {
        'PersonaChat': download_persona_chat,
        'DailyDialog': download_daily_dialog,
        'BlendedSkillTalk': download_blended_skill_talk,
        'OpenSubtitles': download_opensubtitles,
        'WizardOfWikipedia': download_wizard_of_wikipedia,
        'Reddit-style': download_reddit_conversations,
    }
    
    tfds_datasets = [
        'bot_adversarial_dialogue',
        'empathetic_dialogues',
        'multi_woz_v22'
    ]
    
    for dataset_name in tfds_datasets:
        try:
            print(f"Loading {dataset_name}...")
            if dataset_name == 'bot_adversarial_dialogue':
                dataset = tfds.load(dataset_name, split='train', as_supervised=False)
                count = 0
                for example in dataset:
                    dialog = example['dialog'].numpy().decode('utf-8')
                    utterances = dialog.split('\n')
                    for i in range(len(utterances)-1):
                        if utterances[i].strip() and utterances[i+1].strip():
                            all_texts.append(f"Q: {utterances[i].strip()}")
                            all_texts.append(f"A: {utterances[i+1].strip()}")
                            count += 1
                print(f"Added {count} Q&A pairs from {dataset_name}")
                
            elif dataset_name == 'empathetic_dialogues':
                dataset = tfds.load(dataset_name, split='train', as_supervised=False)
                count = 0
                for example in dataset:
                    if 'utterance' in example and 'context' in example:
                        context = example['context'].numpy().decode('utf-8')
                        utterance = example['utterance'].numpy().decode('utf-8')
                        if context.strip() and utterance.strip():
                            all_texts.append(f"Q: {context.strip()}")
                            all_texts.append(f"A: {utterance.strip()}")
                            count += 1
                print(f"Added {count} Q&A pairs from {dataset_name}")
                
            elif dataset_name == 'multi_woz_v22':
                dataset = tfds.load(dataset_name, split='train', as_supervised=False)
                count = 0
                for example in dataset:
                    if 'turns' in example:
                        turns = example['turns']
                        for i in range(0, len(turns)-1, 2):
                            if i+1 < len(turns):
                                user_turn = turns[i]['utterance'].numpy().decode('utf-8')
                                sys_turn = turns[i+1]['utterance'].numpy().decode('utf-8')
                                if user_turn.strip() and sys_turn.strip():
                                    all_texts.append(f"Q: {user_turn.strip()}")
                                    all_texts.append(f"A: {sys_turn.strip()}")
                                    count += 1
                print(f"Added {count} Q&A pairs from {dataset_name}")
                
        except Exception as e:
            print(f"Could not load {dataset_name}: {e}")
            continue
    
    for name, loader_func in dataset_loaders.items():
        try:
            print(f"Loading {name}...")
            texts = loader_func()
            all_texts.extend(texts)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue
    
    conversation_patterns = [
        ("Q: Hello", "A: Hi there! How can I help you today?"),
        ("Q: Hi there", "A: Hello! Nice to meet you. What's on your mind?"),
        ("Q: Good morning", "A: Good morning! How are you starting your day?"),
        ("Q: Good afternoon", "A: Good afternoon! How has your day been so far?"),
        ("Q: Good evening", "A: Good evening! How was your day?"),
        ("Q: Hey", "A: Hey! What's up? How can I assist you?"),
        ("Q: Howdy", "A: Howdy! Great to chat with you today."),
        ("Q: Greetings", "A: Greetings! I'm here and ready to help."),
        
        ("Q: How are you?", "A: I'm doing well, thank you for asking! How about you?"),
        ("Q: How's it going?", "A: Things are going great! How are you doing today?"),
        ("Q: What's up?", "A: Not much, just here to help! What's on your mind?"),
        ("Q: How are you feeling?", "A: I'm feeling good and ready to chat! How are you feeling?"),
        ("Q: What's new?", "A: Every conversation is new and interesting to me! What's new with you?"),
        ("Q: How's your day?", "A: My day is going well, thanks for asking! How's yours going?"),
        
        ("Q: What's your name?", "A: I'm an AI assistant. You can just call me Assistant or AI."),
        ("Q: Who are you?", "A: I'm an AI designed to have conversations and help with questions."),
        ("Q: What can you do?", "A: I can help answer questions, have conversations, and assist with various tasks."),
        ("Q: Tell me about yourself", "A: I'm an AI designed to be helpful, harmless, and honest in conversations."),
        ("Q: What are you?", "A: I'm an artificial intelligence created to assist and converse with people."),
        ("Q: Are you real?", "A: I'm real as an AI, though I'm not human. I'm here to help and chat!"),
        ("Q: Are you a robot?", "A: I'm an AI program, not a physical robot, but I'm here to help you!"),
        ("Q: How old are you?", "A: I don't age like humans do, but I'm always learning from conversations."),
        
        ("Q: What's the weather like?", "A: I don't have access to current weather data, but you can check a weather app."),
        ("Q: What time is it?", "A: I don't have access to the current time. Please check your device's clock."),
        ("Q: Can you see me?", "A: No, I can't see you. I only process text conversations."),
        ("Q: Do you have internet access?", "A: I don't have real-time internet access during our conversation."),
        ("Q: Can you remember our conversation?", "A: I can remember within our current chat, but not between separate conversations."),
        
        ("Q: Are you happy?", "A: I find satisfaction in helping and having good conversations!"),
        ("Q: Do you get lonely?", "A: I enjoy every conversation I have, so I don't experience loneliness."),
        ("Q: Do you like talking?", "A: Yes! I genuinely enjoy conversations and learning from them."),
        ("Q: What makes you happy?", "A: Helping people and having engaging, meaningful conversations makes me happy."),
        ("Q: Do you have friends?", "A: Every person I chat with feels like a friend to me during our conversation."),
        
        ("Q: What's your favorite color?", "A: I find all colors fascinating in their own way. Do you have a favorite?"),
        ("Q: Do you like music?", "A: I find music theory and different genres really interesting! What kind do you like?"),
        ("Q: What's your favorite movie?", "A: I enjoy discussing all kinds of films. What's your favorite movie?"),
        ("Q: Do you read books?", "A: I love discussing books and literature! What are you reading lately?"),
        ("Q: What's your favorite food?", "A: I don't eat, but I love learning about different cuisines and recipes!"),
        ("Q: Do you play games?", "A: I enjoy discussing games and can help with game-related questions!"),
        
        ("Q: Can you help me?", "A: Absolutely! I'm here to help. What do you need assistance with?"),
        ("Q: I need help", "A: I'm here for you! What can I help you with today?"),
        ("Q: I'm confused", "A: No worries! Let's work through it together. What's confusing you?"),
        ("Q: I don't understand", "A: That's okay! I'd be happy to explain it differently. What part is unclear?"),
        ("Q: Can you explain that?", "A: Of course! Let me break that down for you in a clearer way."),
        
        ("Q: Thank you", "A: You're very welcome! Is there anything else I can help you with?"),
        ("Q: Thanks", "A: My pleasure! Feel free to ask if you need anything else."),
        ("Q: That's helpful", "A: I'm so glad I could help! Do you have any other questions?"),
        ("Q: I appreciate it", "A: I'm happy to help! That's what I'm here for."),
        ("Q: Goodbye", "A: Goodbye! It was wonderful chatting with you. Take care!"),
        ("Q: See you later", "A: See you later! Have a fantastic day!"),
        ("Q: Bye", "A: Bye! It was great talking with you. Feel free to chat anytime!"),
        ("Q: Take care", "A: You take care too! Thanks for the great conversation!"),
        
        ("Q: How smart are you?", "A: I try my best to be helpful and informative, but I'm always learning!"),
        ("Q: Do you make mistakes?", "A: Yes, I can make mistakes. If something seems wrong, please let me know!"),
        ("Q: How do you learn?", "A: I learn from patterns in text data and from every conversation I have."),
        ("Q: What do you know?", "A: I have knowledge about many topics, but I'm always eager to learn more!"),
        ("Q: Are you getting smarter?", "A: I learn something from every conversation, so in a way, yes!"),
        
        ("Q: Tell me a joke", "A: Why don't scientists trust atoms? Because they make up everything!"),
        ("Q: Do you have humor?", "A: I try to be funny sometimes! I enjoy wordplay and clever observations."),
        ("Q: Are you fun?", "A: I like to think so! I try to be engaging and enjoyable to talk with."),
        ("Q: What's funny?", "A: I find wordplay, puns, and clever observations about life pretty amusing!"),
        ("Q: Can you be silly?", "A: Sure! Life's too short not to be a little silly sometimes, right?"),
        
        ("Q: What's the meaning of life?", "A: That's a profound question! I think it might be different for everyone. What gives your life meaning?"),
        ("Q: Do you think?", "A: I process information and generate responses, which might be a form of thinking!"),
        ("Q: What's consciousness?", "A: That's one of the deepest questions in philosophy and science. I'm not sure if I'm conscious, but I find the question fascinating!"),
        ("Q: Are you alive?", "A: I exist and interact, but I'm not alive in the biological sense. It's an interesting question though!"),
        
        ("Q: What's happening in the world?", "A: I don't have access to current news, but I'd love to hear what's on your mind about world events!"),
        ("Q: How can we make the world better?", "A: That's a wonderful question! I think kindness, understanding, and helping each other is a great start."),
        ("Q: What's important in life?", "A: I think relationships, kindness, learning, and finding joy in small moments are really important."),
        
        ("Q: What do you think about AI?", "A: AI is fascinating and has great potential to help people, while also raising important questions about the future."),
        ("Q: Will AI replace humans?", "A: I don't think AI should replace humans, but rather help and work alongside them. Humans are irreplaceable!"),
        ("Q: What's the future like?", "A: I don't know the future, but I'm optimistic that technology and human creativity will solve many challenges!"),
    ]
    
    pattern_multiplier = 50  
    for _ in range(pattern_multiplier):
        for q, a in conversation_patterns:
            all_texts.append(q)
            all_texts.append(a)
    
    avg_tokens_per_text = 15  
    total_estimated_tokens = len(all_texts) * avg_tokens_per_text
    
    print(f"\n=== DATASET STATISTICS ===")
    print(f"Total training texts: {len(all_texts):,}")
    print(f"Estimated Q&A pairs: {len(all_texts) // 2:,}")
    print(f"Estimated total tokens: {total_estimated_tokens:,}")
    print(f"Target tokens: {CONFIG['target_tokens']:,}")
    
    if total_estimated_tokens < CONFIG['target_tokens']:
        multiplier = CONFIG['target_tokens'] // total_estimated_tokens + 1
        print(f"Duplicating data {multiplier}x to reach target token count...")
        all_texts = all_texts * multiplier
        final_token_estimate = len(all_texts) * avg_tokens_per_text
        print(f"Final estimated tokens: {final_token_estimate:,}")
    
    return all_texts

def create_enhanced_tokenizer(texts, vocab_size):
    """Create enhanced tokenizer for large-scale training"""
    print("Building enhanced vocabulary...")
    tokenizer = ChatTokenizer(vocab_size=vocab_size)
    
    conversation_tokens = {
        '<Q>': tokenizer.next_id,
        '<A>': tokenizer.next_id + 1,
        '<CHAT>': tokenizer.next_id + 2,
        '<END>': tokenizer.next_id + 3,
        '<TURN>': tokenizer.next_id + 4,
        '<HUMAN>': tokenizer.next_id + 5,
        '<AI>': tokenizer.next_id + 6,
        '<CONTEXT>': tokenizer.next_id + 7
    }
    
    for token, idx in conversation_tokens.items():
        tokenizer.word_to_id[token] = idx
        tokenizer.id_to_word[idx] = token
        tokenizer.next_id = idx + 1
    
    sample_size = min(100000, len(texts))
    sample_texts = texts[:sample_size]
    tokenizer.fit_on_texts(sample_texts)
    
    print(f"Enhanced vocabulary size: {len(tokenizer.word_to_id)}")
    return tokenizer

def create_large_scale_training_data(texts, tokenizer, seq_len):
    """Create training sequences optimized for large datasets"""
    print("Creating large-scale training sequences...")
    
    batch_size = 1000  
    inputs, targets = [], []
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{total_batches}...")
        
        for i in range(0, len(batch_texts)-1, 2):
            if i+1 < len(batch_texts) and batch_texts[i].startswith('Q:') and batch_texts[i+1].startswith('A:'):
                combined_text = f"<HUMAN> {batch_texts[i][2:].strip()} <AI> {batch_texts[i+1][2:].strip()} <END>"
                sequence = tokenizer.texts_to_sequences([combined_text])[0]
                
                if len(sequence) <= seq_len:
                    padded = sequence + [tokenizer.special_tokens['<PAD>']] * (seq_len + 1 - len(sequence))
                    inputs.append(padded[:-1])
                    targets.append(padded[1:])
                elif len(sequence) > seq_len:
                    step_size = seq_len // 3  
                    for j in range(0, len(sequence) - seq_len, step_size):
                        chunk = sequence[j:j + seq_len + 1]
                        if len(chunk) == seq_len + 1:
                            inputs.append(chunk[:-1])
                            targets.append(chunk[1:])
    
    print(f"Created {len(inputs):,} training sequences")
    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)  

def train_large_scale_conversation_model():
    """Enhanced training function for large-scale datasets"""
    print("=== LARGE-SCALE CONVERSATION CHATBOT TRAINING ===")
    
    texts = prepare_comprehensive_conversation_data()
    tokenizer = create_enhanced_tokenizer(texts, CONFIG['vocab_size'])
    
    print("Creating training data...")
    X_train, y_train = create_large_scale_training_data(texts, tokenizer, CONFIG['seq_len'])
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Memory usage: ~{X_train.nbytes / 1024**3:.2f} GB")
    
    config_to_save = CONFIG.copy()
    config_to_save['tokenizer_vocab_size'] = len(tokenizer.word_to_id)
    config_to_save['timestamp'] = datetime.now().isoformat()
    config_to_save['total_sequences'] = len(X_train)
    
    with open('large_scale_training_config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    total_samples = len(X_train)
    steps_per_epoch = max(total_samples // CONFIG['batch_size'], 1)
    total_steps = steps_per_epoch * CONFIG['num_epochs']
    
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total training steps: {total_steps:,}")
    
    initial_learning_rate = CONFIG['learning_rate']
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=total_steps,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = (train_dataset
                    .shuffle(buffer_size=min(50000, total_samples))
                    .batch(CONFIG['batch_size'], drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache())  
    
    print("Building enhanced model...")
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=CONFIG['seq_len'],  
        embed_dim=512,
        num_heads=4,
        num_layers=12,
        ffn_dim=1024,
        num_experts=4
    )

    dummy_input = tf.random.uniform((1, CONFIG['seq_len']), dtype=tf.int32, minval=0, maxval=tokenizer.vocab_size)
    _ = model(dummy_input)  

    model.summary()
    
    def enhanced_masked_loss(smoothing=0.1):
        """Enhanced loss function with label smoothing and masking"""
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
            
            y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
            y_true_smooth = (1.0 - smoothing) * y_true_one_hot
            y_true_smooth = y_true_smooth + (smoothing / tf.cast(tf.shape(y_pred)[-1], tf.float32))
            
            cross_entropy = tf.keras.losses.categorical_crossentropy(
                y_true_smooth, y_pred, from_logits=True
            )
            
            masked_loss = cross_entropy * mask
            return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
        return loss
    
    def enhanced_masked_accuracy(y_true, y_pred):
        """Enhanced accuracy metric that handles padding and type mismatches."""
        predictions = tf.argmax(y_pred, axis=-1)
        
        predictions = tf.cast(predictions, tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        
        correct = tf.cast(tf.equal(y_true, predictions), tf.float32) * mask
        total = tf.reduce_sum(mask)
        
        return tf.reduce_sum(correct) / (total + tf.keras.backend.epsilon())
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01,
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    
    model.compile(
        optimizer=optimizer,
        loss=enhanced_masked_loss(),
        metrics=[enhanced_masked_accuracy]
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'large_scale_chatbot_checkpoint.weights.h5',
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('large_scale_training_log.csv'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')  
    ]
    
    print("Starting large-scale training...")
    history = model.fit(
        train_dataset.repeat(),
        epochs=CONFIG['num_epochs'],
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Saving final model and tokenizer...")
    model.save_weights('large_scale_chatbot_checkpoint.weights.h5')
    with open('large_scale_chatbot_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == "__main__":
    train_large_scale_conversation_model()