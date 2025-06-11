import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import os
import logging
import time

# Optional: only import if transformers is available
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: transformers library not found. Tokenizer features will be disabled.")
    HAS_TRANSFORMERS = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the MiniGPT model."""
    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        use_custom_attention: bool = True,
        use_rotary_embeddings: bool = True,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        seq_len: int = 1024
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_custom_attention = use_custom_attention
        self.use_rotary_embeddings = use_rotary_embeddings
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seq_len = seq_len

class RotaryPositionalEmbedding(layers.Layer):
    """Rotary Positional Embedding layer."""
    def __init__(self, dim, max_seq_len=1024, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create position indices
        position = tf.range(max_seq_len, dtype=tf.float32)
        div_term = tf.exp(tf.range(0, dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / dim))
        
        # Calculate sin and cos values
        pos = tf.expand_dims(position, 1) * tf.expand_dims(div_term, 0)
        self.sin_vals = tf.sin(pos)
        self.cos_vals = tf.cos(pos)
    
    def _apply_rotary_pos_emb(self, x, sin, cos):
        """Apply rotary positional embeddings to input tensor."""
        # Reshape input for broadcasting
        x_shape = tf.shape(x)
        batch_size = x_shape[0]
        seq_len = x_shape[1]
        
        # Reshape sin and cos for broadcasting
        sin = tf.expand_dims(sin[:seq_len], 0)  # [1, seq_len, dim/2]
        cos = tf.expand_dims(cos[:seq_len], 0)  # [1, seq_len, dim/2]
        
        # Split input into real and imaginary parts
        x1, x2 = tf.split(x, 2, axis=-1)
        
        # Apply rotary embeddings
        rotated_x1 = cos * x1 - sin * x2
        rotated_x2 = sin * x1 + cos * x2
        
        # Concatenate rotated parts
        return tf.concat([rotated_x1, rotated_x2], axis=-1)
    
    def call(self, x, seq_len=None):
        """Apply rotary positional embeddings."""
        if seq_len is None:
            seq_len = tf.shape(x)[1]
        
        # Get sin and cos values for the sequence length
        sin = self.sin_vals[:seq_len]
        cos = self.cos_vals[:seq_len]
        
        # Apply rotary embeddings
        return self._apply_rotary_pos_emb(x, sin, cos)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'max_seq_len': self.max_seq_len
        })
        return config

class CustomMultiHeadAttention(tf.keras.layers.Layer):
    """Custom multi-head attention layer with optional rotary embeddings."""
    
    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        use_rotary_embeddings: bool = True,
        dropout: float = 0.1,
        name: str = None,
        **kwargs
    ):
        # Remove custom arguments from kwargs before passing to parent
        custom_kwargs = {
            'use_rotary_embeddings': use_rotary_embeddings
        }
        for key in custom_kwargs:
            kwargs.pop(key, None)
            
        super().__init__(name=name, **kwargs)
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.use_rotary_embeddings = use_rotary_embeddings
        self.dropout = dropout
        
        # Create query, key, value projection layers
        self.query_dense = tf.keras.layers.Dense(num_heads * key_dim)
        self.key_dense = tf.keras.layers.Dense(num_heads * key_dim)
        self.value_dense = tf.keras.layers.Dense(num_heads * key_dim)
        
        # Create output projection layer
        self.output_dense = tf.keras.layers.Dense(num_heads * key_dim)
        
        # Create dropout layer
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        
        # Create rotary embeddings if enabled
        if use_rotary_embeddings:
            self.rotary_embeddings = RotaryPositionalEmbedding(key_dim)
    
    def call(self, inputs, attention_mask=None, training=False):
        # Get input shape
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Project inputs to query, key, value
        q = self.query_dense(inputs)  # [batch_size, seq_len, num_heads * key_dim]
        k = self.key_dense(inputs)    # [batch_size, seq_len, num_heads * key_dim]
        v = self.value_dense(inputs)  # [batch_size, seq_len, num_heads * key_dim]
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.key_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.key_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.key_dim])
        
        # Apply rotary embeddings if enabled
        if self.use_rotary_embeddings:
            q = self.rotary_embeddings(q)
            k = self.rotary_embeddings(k)
        
        # Transpose for attention computation
        q = tf.transpose(q, [0, 2, 1, 3])  # [batch_size, num_heads, seq_len, key_dim]
        k = tf.transpose(k, [0, 2, 1, 3])  # [batch_size, num_heads, seq_len, key_dim]
        v = tf.transpose(v, [0, 2, 1, 3])  # [batch_size, num_heads, seq_len, key_dim]
        
        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True)  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / tf.math.sqrt(tf.cast(self.key_dim, scores.dtype))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout_layer(attention_weights, training=training)
        
        # Apply attention weights to values
        context = tf.matmul(attention_weights, v)  # [batch_size, num_heads, seq_len, key_dim]
        
        # Transpose and reshape for output
        context = tf.transpose(context, [0, 2, 1, 3])  # [batch_size, seq_len, num_heads, key_dim]
        context = tf.reshape(context, [batch_size, seq_len, self.num_heads * self.key_dim])
        
        # Project to output
        output = self.output_dense(context)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'use_rotary_embeddings': self.use_rotary_embeddings,
            'dropout': self.dropout
        })
        return config

class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer with custom attention implementation"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_seq_len=2048, use_custom_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_custom_attention = use_custom_attention
        
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections
        self.query_proj = layers.Dense(embed_dim, name='query_proj')
        self.key_proj = layers.Dense(embed_dim, name='key_proj')
        self.value_proj = layers.Dense(embed_dim, name='value_proj')
        
        # Output projection
        self.output_proj = layers.Dense(embed_dim, name='output_proj')
        
        # Rotary position embedding
        self.rope = RotaryPositionalEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            name='rope'
        )
        
        # Custom Multi-Head Attention layer
        if use_custom_attention:
            self.custom_attention = CustomMultiHeadAttention(
                num_heads=num_heads,
                key_dim=self.head_dim,
                dropout=dropout,
                name='custom_attention'
            )
        
        # Dropout
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        if self.use_custom_attention:
            # Use custom attention - simplified mask creation
            attention = self.custom_attention(x, use_causal_mask=True, training=training)
            return attention
        else:
            # Standard attention with RoPE
            # Get Q, K, V
            q = self.query_proj(x)
            k = self.key_proj(x)
            v = self.value_proj(x)
            
            # Reshape for multi-head attention
            q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
            k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
            v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
            
            # Apply RoPE to queries and keys
            q = self.rope(q, seq_len)
            k = self.rope(k, seq_len)
            
            # Transpose to (batch_size, num_heads, seq_len, head_dim)
            q = tf.transpose(q, perm=[0, 2, 1, 3])
            k = tf.transpose(k, perm=[0, 2, 1, 3])
            v = tf.transpose(v, perm=[0, 2, 1, 3])
            
            # Create causal mask
            mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            mask = tf.where(mask == 0, -1e9, 0.0)
            mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)  # [1, 1, seq_len, seq_len]
            
            # Standard attention
            attention = self._scaled_dot_product_attention(q, k, v, mask, training)
            
            # Transpose back and reshape
            attention = tf.transpose(attention, perm=[0, 2, 1, 3])
            attention = tf.reshape(attention, (batch_size, seq_len, self.embed_dim))
            
            # Output projection
            output = self.output_proj(attention)
            
            return output
    
    def _scaled_dot_product_attention(self, q, k, v, mask=None, training=False):
        """Standard scaled dot-product attention (fallback)"""
        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores += mask
        
        # Apply softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, v)
        
        return output

class FeedForward(layers.Layer):
    """Feed-forward network"""
    
    def __init__(self, embed_dim, ffn_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        
        self.dense1 = layers.Dense(ffn_dim, activation='gelu', name='dense1')
        self.dense2 = layers.Dense(embed_dim, name='dense2')
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with optional custom attention and rotary embeddings."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        use_custom_attention: bool = True,
        use_rotary_embeddings: bool = True,
        name: str = None,
        **kwargs
    ):
        # Remove custom arguments from kwargs before passing to parent
        custom_kwargs = {
            'use_custom_attention': use_custom_attention,
            'use_rotary_embeddings': use_rotary_embeddings
        }
        for key in custom_kwargs:
            kwargs.pop(key, None)
            
        super().__init__(name=name, **kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.use_custom_attention = use_custom_attention
        self.use_rotary_embeddings = use_rotary_embeddings
        
        # Create layers
        if use_custom_attention:
            self.attention = CustomMultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                use_rotary_embeddings=use_rotary_embeddings
            )
        else:
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads
            )
            
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, mask=None, training=False):
        # Apply attention
        attention_output = self.attention(
            inputs, inputs,
            attention_mask=mask,
            training=training
        )
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)
        
        # Apply feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'use_custom_attention': self.use_custom_attention,
            'use_rotary_embeddings': self.use_rotary_embeddings
        })
        # Multi-head attention with RoPE
        self.attention = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            max_seq_len=2048,
            use_custom_attention=self.use_custom_attention,
            name='attention'
        )
        
        # Feed-forward network
        self.ffn = FeedForward(
            embed_dim=self.embed_dim,
            ffn_dim=self.ff_dim,
            dropout=self.dropout_rate,
            name='ffn'
        )
        
        # Layer normalization
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name='ln1')
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name='ln2')
        
        # Dropout
        self.dropout = layers.Dropout(self.dropout_rate)
        
        return config

class EnhancedMiniGPT(tf.keras.Model):
    """Enhanced MiniGPT model with improved architecture and features."""
    
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(name="enhanced_minigpt", **kwargs)
        self.config = config
        self.batch_size = 2  # Set default batch size to 2
        
        # Token and positional embeddings
        self.token_embedding = tf.keras.layers.Embedding(
            config.vocab_size,
            config.embed_dim,
            name="token_embedding"
        )
        self.position_embedding = tf.keras.layers.Embedding(
            config.max_seq_len,
            config.embed_dim,
            name="position_embedding"
        )
        
        # Transformer blocks
        self.transformer_blocks = []
        for i in range(config.num_layers):
            block = TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_dim=config.ffn_dim,
                dropout_rate=config.dropout,
                use_custom_attention=True,
                use_rotary_embeddings=True,
                name=f"transformer_block_{i}"
            )
            self.transformer_blocks.append(block)
        
        # Output layers
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = tf.keras.layers.Dense(
            config.vocab_size,
            name="output_layer"
        )
        
        # Initialize tokenizer if available
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        except ImportError:
            logger.warning("transformers library not available, tokenizer not initialized")
            self.tokenizer = None
    
    def call(self, inputs, training=False, mask=None):
        # Get input shape and ensure batch size is 2
        batch_size = tf.shape(inputs)[0]
        if batch_size != 2:
            logger.warning(f"Expected batch size 2, got {batch_size}. Adjusting input.")
            inputs = tf.slice(inputs, [0, 0], [2, -1])
            batch_size = 2
            
        seq_len = tf.shape(inputs)[1]
        
        # Create position indices
        positions = tf.range(0, seq_len, dtype=tf.int32)
        positions = tf.expand_dims(positions, 0)
        positions = tf.tile(positions, [batch_size, 1])
        
        # Get embeddings
        token_embeddings = self.token_embedding(inputs)
        position_embeddings = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=mask, training=training)
        
        # Final layer norm and output
        x = self.layernorm(x)
        logits = self.output_layer(x)
        
        return logits
    
    def generate(self, input_ids, max_length: int = 100, temperature: float = 0.7,
                top_k: int = 50, top_p: float = 0.9, batch_size: int = 2):
        """Generate text using the model."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
            
        # Ensure batch size is 2
        if batch_size != 2:
            logger.warning(f"Adjusting batch size from {batch_size} to 2")
            batch_size = 2
            
        # Initialize generation
        generated = tf.identity(input_ids)
        current_length = tf.shape(input_ids)[1]
        
        # Generate tokens
        for _ in range(max_length - current_length):
            # Get model predictions
            logits = self(generated, training=False)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < tf.math.top_k(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = tf.where(indices_to_remove, tf.ones_like(next_token_logits) * -float('inf'), next_token_logits)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits = tf.sort(next_token_logits, direction='DESCENDING')
                sorted_indices = tf.argsort(next_token_logits, direction='DESCENDING')
                cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = tf.concat([tf.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, :-1]], axis=-1)
                indices_to_remove = tf.batch_gather(sorted_indices_to_remove, sorted_indices)
                next_token_logits = tf.where(indices_to_remove, tf.ones_like(next_token_logits) * -float('inf'), next_token_logits)
            
            # Sample next token
            probs = tf.nn.softmax(next_token_logits, axis=-1)
            next_token = tf.random.categorical(probs, num_samples=1)
            
            # Append to generated sequence
            generated = tf.concat([generated, next_token], axis=1)
            
            # Check for end of sequence token
            if tf.reduce_any(next_token == self.tokenizer.eos_token_id):
                break
        
        return generated
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'config': self.config,
            'batch_size': self.batch_size
        })
        return config

    def generate_text(self, prompt: str, max_length: int = 512, batch_size: int = 8):
        """Generate text from a prompt."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Please install transformers library.")
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='tf')
        input_ids = tf.cast(input_ids, tf.int32)
        
        # Generate tokens
        for _ in range(max_length):
            # Get model predictions
            logits = self(input_ids, training=False)
            
            # Get next token probabilities
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature and sample
            temperature = 0.7
            next_token_logits = next_token_logits / temperature
            next_token_probs = tf.nn.softmax(next_token_logits, axis=-1)
            
            # Sample next token
            next_token = tf.random.categorical(next_token_probs, num_samples=1)
            next_token = tf.cast(next_token, tf.int32)
            
            # Append to input_ids
            input_ids = tf.concat([input_ids, next_token], axis=1)
            
            # Stop if we predict the end of sequence token
            if next_token[0, 0] == self.tokenizer.eos_token_id:
                break
        
        # Decode and return generated text
        generated_text = self.tokenizer.decode(input_ids[0].numpy())
        return generated_text
        
    def chat(self, message: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Chat with the model (requires tokenizer)"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available. Please install transformers package.")
            
        # Format message
        prompt = f"Human: {message}\nAI:"
        
        # Generate response
        response = self.generate_text(prompt, max_length=max_length, temperature=temperature)
        
        # Extract AI response
        try:
            response = response.split("AI:")[-1].strip()
        except:
            response = response.strip()
            
        return response
        
    def compute_loss(self, input_ids, labels=None):
        """Compute the loss for training."""
        if labels is None:
            labels = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
            
        # Get model predictions
        logits = self(input_ids, training=True)
        
        # Compute loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        
        return tf.reduce_mean(loss)

    def benchmark_speed(self, batch_sizes=[1, 2, 4, 8], seq_lengths=[128, 256, 512]) -> Dict:
        """Benchmark model speed with different batch sizes and sequence lengths."""
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                key = f"batch_{batch_size}_seq_{seq_len}"
                
                # Create dummy input
                input_ids = tf.random.uniform(
                    (batch_size, seq_len),
                    minval=0,
                    maxval=self.config.vocab_size,
                    dtype=tf.int32
                )
                
                # Warmup
                for _ in range(3):
                    _ = self(input_ids, training=False)
                
                # Benchmark
                num_runs = 10
                start_time = time.time()
                
                for _ in range(num_runs):
                    _ = self(input_ids, training=False)
                
                elapsed = time.time() - start_time
                
                # Calculate metrics
                tokens_per_second = (batch_size * seq_len * num_runs) / elapsed
                
                results[key] = {
                    "tokens_per_second": tokens_per_second,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "elapsed_time": elapsed
                }
        
        return results