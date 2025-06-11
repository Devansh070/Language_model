import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import os
import logging

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
    """Configuration for the MiniGPT model"""
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    max_seq_len: int = 512
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    ffn_dim: int = 2048
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    use_custom_attention: bool = True  # Enable Custom Multi-Head Attention by default

class RotaryPositionalEmbedding(layers.Layer):
    """Rotary Positional Embedding (RoPE)"""
    
    def __init__(self, dim, max_seq_len=2048, base=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
    def build(self, input_shape):
        # Pre-compute frequency components
        inv_freq = 1.0 / (self.base ** (tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim))
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=inv_freq.shape,
            initializer='zeros',
            trainable=False
        )
        self.inv_freq.assign(inv_freq)
        super().build(input_shape)
        
    def call(self, x, seq_len=None):
        """
        Apply rotary position embedding to input tensor
        x: [batch_size, seq_len, num_heads, head_dim]
        """
        if seq_len is None:
            seq_len = tf.shape(x)[1]
        
        # Create position indices
        position = tf.range(seq_len, dtype=tf.float32)
        position = tf.expand_dims(position, 1)  # [seq_len, 1]
        
        # Compute frequencies
        freqs = tf.matmul(position, tf.expand_dims(self.inv_freq, 0))  # [seq_len, dim//2]
        
        # Create sin and cos
        sin_vals = tf.sin(freqs)  # [seq_len, dim//2]
        cos_vals = tf.cos(freqs)  # [seq_len, dim//2]
        
        # Expand dimensions for broadcasting
        sin_vals = tf.expand_dims(tf.expand_dims(sin_vals, 0), 2)  # [1, seq_len, 1, dim//2]
        cos_vals = tf.expand_dims(tf.expand_dims(cos_vals, 0), 2)  # [1, seq_len, 1, dim//2]
        
        # Apply rotary embedding
        x_rotated = self._apply_rotary_pos_emb(x, sin_vals, cos_vals)
        
        return x_rotated
    
    def _apply_rotary_pos_emb(self, x, sin, cos):
        """Apply the actual rotary transformation"""
        # Split x into two halves
        x1 = x[..., ::2]   # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Apply rotation
        # cos * x + sin * rotate(x) where rotate swaps x1 and x2 with sign flip
        rotated_x1 = cos * x1 - sin * x2
        rotated_x2 = sin * x1 + cos * x2
        
        # Interleave back
        rotated = tf.stack([rotated_x1, rotated_x2], axis=-1)
        rotated = tf.reshape(rotated, tf.shape(x))
        
        return rotated

class CustomMultiHeadAttention(layers.Layer):
    """
    Custom implementation of Multi-Head Attention mechanism from scratch.
    Supports causal masking and padding masking.
    """
    
    def __init__(self, num_heads, key_dim, value_dim=None, dropout=0.0, use_bias=True, 
                 output_shape=None, attention_axes=None, kernel_initializer="glorot_uniform",
                 bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.dropout = dropout
        self.use_bias = use_bias
        self.output_shape = output_shape
        self.attention_axes = attention_axes
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
    
    def build(self, input_shape):
        # Get input dimensions
        if isinstance(input_shape, list):
            query_shape = input_shape[0]
        else:
            query_shape = input_shape
        
        # Get embedding dimension
        self.embed_dim = query_shape[-1]
        
        # Validate embedding dimension
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding dimension {self.embed_dim} must be divisible by number of heads {self.num_heads}"
            )
        
        # Calculate dimensions for each head
        self.head_dim = self.embed_dim // self.num_heads
        
        # Create projection matrices for Q, K, V
        self.query_dense = layers.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="query"
        )
        
        self.key_dense = layers.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="key"
        )
        
        self.value_dense = layers.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="value"
        )
        
        # Output projection
        self.output_dense = layers.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="output"
        )
        
        # Dropout layer
        self.dropout_layer = layers.Dropout(self.dropout)
        
        super().build(input_shape)
    
    def call(self, inputs, mask=None, use_causal_mask=False, training=None):
        """
        Compute multi-head attention.
        
        Args:
            inputs: Query tensor of shape [batch_size, seq_len, embed_dim]
                   or list of [query, key, value] tensors
            mask: Padding mask of shape [batch_size, seq_len] or [batch_size, seq_len, seq_len]
            use_causal_mask: Whether to use causal masking
            training: Whether the layer is in training mode
        
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Handle different input formats
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        
        # Get shapes
        batch_size = tf.shape(query)[0]
        query_len = tf.shape(query)[1]
        key_len = tf.shape(key)[1]
        
        # Project inputs to Q, K, V
        query = self.query_dense(query)  # [batch_size, query_len, embed_dim]
        key = self.key_dense(key)        # [batch_size, key_len, embed_dim]
        value = self.value_dense(value)  # [batch_size, key_len, embed_dim]
        
        # Reshape for multi-head attention
        query = tf.reshape(query, [batch_size, query_len, self.num_heads, self.head_dim])
        key = tf.reshape(key, [batch_size, key_len, self.num_heads, self.head_dim])
        value = tf.reshape(value, [batch_size, key_len, self.num_heads, self.head_dim])
        
        # Transpose for attention computation
        query = tf.transpose(query, [0, 2, 1, 3])  # [batch_size, num_heads, query_len, head_dim]
        key = tf.transpose(key, [0, 2, 1, 3])      # [batch_size, num_heads, key_len, head_dim]
        value = tf.transpose(value, [0, 2, 1, 3])  # [batch_size, num_heads, key_len, head_dim]
        
        # Compute attention scores
        scores = tf.matmul(query, key, transpose_b=True)  # [batch_size, num_heads, query_len, key_len]
        
        # Scale scores
        scores = scores / tf.sqrt(tf.cast(self.head_dim, scores.dtype))
        
        # Apply masks
        if mask is not None:
            # Expand mask for broadcasting
            if len(mask.shape) == 2:
                mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
            else:
                mask = tf.expand_dims(mask, 1)
            
            # Convert mask to attention mask
            attention_mask = (1.0 - tf.cast(mask, scores.dtype)) * -10000.0
            scores = scores + attention_mask
        
        # Apply causal mask if requested
        if use_causal_mask:
            # Create causal mask
            causal_mask = tf.linalg.band_part(
                tf.ones((query_len, key_len)), -1, 0
            )
            causal_mask = tf.cast(causal_mask, scores.dtype)
            causal_mask = tf.expand_dims(tf.expand_dims(causal_mask, 0), 0)
            
            # Apply causal mask
            scores = scores * causal_mask + (1.0 - causal_mask) * -10000.0
        
        # Compute attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout_layer(attention_weights, training=training)
        
        # Compute attention output
        attention_output = tf.matmul(attention_weights, value)  # [batch_size, num_heads, query_len, head_dim]
        
        # Transpose and reshape
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])  # [batch_size, query_len, num_heads, head_dim]
        attention_output = tf.reshape(attention_output, [batch_size, query_len, self.embed_dim])
        
        # Final projection
        output = self.output_dense(attention_output)
        
        return output
    
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape = input_shape[0]
        else:
            query_shape = input_shape
        
        return tf.TensorShape([*query_shape[:-1], self.embed_dim])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "dropout": self.dropout,
            "use_bias": self.use_bias,
            "output_shape": self.output_shape,
            "attention_axes": self.attention_axes,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "activity_regularizer": self.activity_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_constraint": self.bias_constraint,
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
            # For custom attention, pass the original x and let it handle projections
            # Create causal mask for custom attention
            mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            mask = tf.where(mask == 0, -1e9, 0.0)
            mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)  # [1, 1, seq_len, seq_len]
            
            attention = self.custom_attention(x, mask=mask, use_causal_mask=True, training=training)
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

class TransformerBlock(layers.Layer):
    """Transformer block with attention and feed-forward layers"""
    
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, layer_norm_epsilon=1e-5, max_seq_len=2048, **kwargs):
        super().__init__(**kwargs)
        
        # Multi-head attention with RoPE
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_custom_attention=True,
            name='attention'
        )
        
        # Feed-forward network
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            name='ffn'
        )
        
        # Layer normalization
        self.ln1 = layers.LayerNormalization(epsilon=layer_norm_epsilon, name='ln1')
        self.ln2 = layers.LayerNormalization(epsilon=layer_norm_epsilon, name='ln2')
        
        # Dropout
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x, training=False):
        # Pre-norm attention
        attn_input = self.ln1(x)
        attn_output = self.attention(attn_input, training=training)
        attn_output = self.dropout(attn_output, training=training)
        x = x + attn_output
        
        # Pre-norm feed-forward
        ffn_input = self.ln2(x)
        ffn_output = self.ffn(ffn_input, training=training)
        ffn_output = self.dropout(ffn_output, training=training)
        x = x + ffn_output
        
        return x

class EnhancedMiniGPT(Model):
    """Enhanced MiniGPT model with improved architecture and optimizations"""
    
    def __init__(self, config: ModelConfig = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or ModelConfig()
        
        # Initialize tokenizer if available
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # Token embeddings (no position embeddings needed with RoPE)
        self.token_embedding = layers.Embedding(
            self.config.vocab_size,
            self.config.embed_dim,
            name='token_embedding'
        )
        
        # Transformer blocks with RoPE
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                ffn_dim=self.config.ffn_dim,
                dropout=self.config.dropout,
                layer_norm_epsilon=self.config.layer_norm_epsilon,
                max_seq_len=self.config.max_seq_len,
                name=f'transformer_block_{i}'
            ) for i in range(self.config.num_layers)
        ]
        
        # Final layer normalization
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=self.config.layer_norm_epsilon,
            name='final_layer_norm'
        )
        
        # Output layer (language modeling head)
        self.lm_head = layers.Dense(
            self.config.vocab_size,
            use_bias=False,
            name='lm_head'
        )
        
        # Dropout
        self.dropout = layers.Dropout(self.config.dropout)
        
    def call(self, inputs, training=False):
        """Forward pass with RoPE (no position embeddings needed)"""
        # Get token embeddings only
        x = self.token_embedding(inputs)
        
        # Apply dropout
        x = self.dropout(x, training=training)
        
        # Apply transformer blocks (RoPE is applied inside attention)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Apply final layer normalization
        x = self.final_layer_norm(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        return logits
    
    def build_model(self):
        """Build the model by calling it with dummy input"""
        dummy_input = tf.zeros((1, self.config.max_seq_len), dtype=tf.int32)
        _ = self(dummy_input)
        logger.info(f"Model built successfully with {self.count_params():,} parameters")
    
    def generate(self, input_ids, max_length: int = 100, temperature: float = 0.7, 
                 top_k: int = 50, top_p: float = 0.9, pad_token_id: int = 0):
        """Generate text continuation"""
        # Ensure input is tensor
        if not isinstance(input_ids, tf.Tensor):
            input_ids = tf.constant(input_ids, dtype=tf.int32)
        
        # Ensure batch dimension
        if len(input_ids.shape) == 1:
            input_ids = tf.expand_dims(input_ids, 0)
        
        batch_size = tf.shape(input_ids)[0]
        current_length = tf.shape(input_ids)[1]
        
        # Generate tokens
        for _ in range(max_length):
            # Truncate input if too long
            if current_length >= self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len+1:]
                current_length = tf.shape(input_ids)[1]
            
            # Get model predictions
            logits = self(input_ids, training=False)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = tf.nn.top_k(next_token_logits, k=min(top_k, tf.shape(next_token_logits)[-1]))
                next_token_logits = tf.where(
                    next_token_logits < tf.reduce_min(top_k_logits, axis=-1, keepdims=True),
                    -float('inf'),
                    next_token_logits
                )
            
            # Sample next token
            next_token = tf.random.categorical(next_token_logits, num_samples=1, dtype=tf.int32)
            
            # Append to input
            input_ids = tf.concat([input_ids, next_token], axis=1)
            current_length += 1
            
            # Check for pad token (simple stopping condition)
            if tf.reduce_all(next_token == pad_token_id):
                break
        
        return input_ids
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7):
        """Generate text from a string prompt (requires tokenizer)"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not available. Install transformers library or use generate() with token IDs.")
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='tf')
        
        # Generate
        generated_ids = self.generate(input_ids, max_length, temperature)
        
        # Decode and return
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    def chat(self, message: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Chat with the model (requires tokenizer)"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not available. Install transformers library.")
        
        # Format input
        prompt = f"Human: {message}\nAssistant:"
        
        # Generate response
        response = self.generate_text(prompt, max_length, temperature)
        
        # Extract assistant response
        try:
            response = response.split("Assistant:")[-1].strip()
        except:
            response = response.strip()
        
        return response
    
    def compute_loss(self, input_ids, labels=None):
        """Compute language modeling loss"""
        if labels is None:
            # For language modeling, labels are input_ids shifted by one
            labels = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
        
        # Get logits
        logits = self(input_ids, training=True)
        
        # Compute loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        
        # Apply padding mask if needed
        mask = tf.cast(tf.not_equal(labels, 0), tf.float32)
        loss = loss * mask
        
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)