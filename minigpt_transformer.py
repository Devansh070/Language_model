import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Set mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional: only import if transformers is available
try:
    from transformers import GPT2Tokenizer as _GPT2Tokenizer
    from transformers import AutoTokenizer as _AutoTokenizer
    GPT2TokenizerClass = _GPT2Tokenizer
    HAS_TRANSFORMERS = True
    logger.info("Successfully imported transformers library")
except ImportError:
    logger.warning("transformers library not found. Tokenizer features will be disabled.")
    HAS_TRANSFORMERS = False
    GPT2Tokenizer = None
    AutoTokenizer = None

def create_dense_float16(units, activation=None, name=None, **kwargs):
    """Helper function to create Dense layers with float16 dtype."""
    return tf.keras.layers.Dense(
        units=units,
        activation=activation,
        dtype=tf.float16,
        name=name,
        **kwargs
    )

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
        batch_size: int = 1,
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
    """Fixed Rotary Positional Embedding layer with proper float16 support."""
    
    def __init__(self, dim, max_seq_len=1024, dtype=tf.float16, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute in float32 for precision, store as variables for gradient flow
        with tf.name_scope('rotary_embedding_init'):
            position = tf.range(max_seq_len, dtype=tf.float32)
            div_term_exp = tf.range(0, dim, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / dim)
            div_term = tf.exp(div_term_exp)
            
            # Compute positional encodings in float32
            pos = tf.einsum('i,j->ij', position, div_term)
            
            # Create as variables instead of constants for better gradient flow
            self.sin_vals = self.add_weight(
                name="sin_vals",
                shape=(max_seq_len, dim // 2),
                initializer="zeros",
                trainable=False,
                dtype=tf.float16
            )
            self.cos_vals = self.add_weight(
                name="cos_vals", 
                shape=(max_seq_len, dim // 2),
                initializer="zeros",
                trainable=False,
                dtype=tf.float16
            )
            
            # Assign computed values
            self.sin_vals.assign(tf.cast(tf.sin(pos), tf.float16))
            self.cos_vals.assign(tf.cast(tf.cos(pos), tf.float16))
    
    def _apply_rotary_pos_emb(self, x, sin, cos):
        """Fixed rotary positional embeddings with proper broadcasting."""
        # Get shapes
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1] 
        num_heads = tf.shape(x)[2]
        head_dim = tf.shape(x)[3]
        
        # Slice sin/cos to match sequence length
        sin = sin[:seq_len, :]  # [seq_len, head_dim//2]
        cos = cos[:seq_len, :]  # [seq_len, head_dim//2]
        
        # Expand dimensions for broadcasting: [1, seq_len, 1, head_dim//2]
        sin = tf.expand_dims(tf.expand_dims(sin, 0), 2)
        cos = tf.expand_dims(tf.expand_dims(cos, 0), 2)
        
        # Split x into two halves
        x1, x2 = tf.split(x, 2, axis=-1)
        
        # Apply rotation
        rotated_x1 = cos * x1 - sin * x2
        rotated_x2 = sin * x1 + cos * x2
        
        # Concatenate back
        result = tf.concat([rotated_x1, rotated_x2], axis=-1)
        return result
    
    def call(self, x, seq_len=None):
        """Apply rotary positional embeddings."""
        if seq_len is None:
            seq_len = tf.shape(x)[1]
        
        # Ensure we don't exceed precomputed values
        seq_len = tf.minimum(seq_len, self.max_seq_len)
        
        return self._apply_rotary_pos_emb(x, self.sin_vals, self.cos_vals)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'max_seq_len': self.max_seq_len,
            'dtype': self.dtype
        })
        return config

class CustomMultiHeadAttention(tf.keras.layers.Layer):
    """Fixed custom multi-head attention layer with proper float16 handling."""
    
    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        use_rotary_embeddings: bool = True,
        dropout: float = 0.1,
        name: Optional[str] = "custom_attention",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.use_rotary_embeddings = use_rotary_embeddings
        self.dropout = dropout
        
        # Create projection layers - let mixed precision handle dtype
        self.query_dense = layers.Dense(num_heads * key_dim, dtype=tf.float16, name='query_dense')
        self.key_dense = layers.Dense(num_heads * key_dim, dtype=tf.float16, name='key_dense')
        self.value_dense = layers.Dense(num_heads * key_dim, dtype=tf.float16, name='value_dense')
        self.output_dense = layers.Dense(num_heads * key_dim, dtype=tf.float16, name='output_dense')
        
        self.dropout_layer = tf.keras.layers.Dropout(dropout, dtype=tf.float16)
        
        # Create rotary embeddings
        if use_rotary_embeddings:
            self.rotary_embeddings = RotaryPositionalEmbedding(key_dim)
        
        # Store scaling factor as a variable
        self.scale = self.add_weight(
            name="attention_scale",
            shape=(),
            initializer="zeros",
            trainable=False,
            dtype=tf.float16
        )
        self.scale.assign(tf.cast(1.0 / tf.math.sqrt(float(key_dim)), tf.float16))
    
    def call(self, inputs, attention_mask=None, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Project to Q, K, V
        q = self.query_dense(inputs)
        k = self.key_dense(inputs)
        v = self.value_dense(inputs)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.key_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.key_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.key_dim])
        
        # Apply rotary embeddings
        if self.use_rotary_embeddings:
            q = self.rotary_embeddings(q)
            k = self.rotary_embeddings(k)
        
        # Transpose for attention computation
        q = tf.transpose(q, [0, 2, 1, 3])  # [batch, heads, seq_len, key_dim]
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        q = tf.cast(self.query_dense(inputs), tf.float16)
        k = tf.cast(self.key_dense(inputs), tf.float16)
        v = tf.cast(self.value_dense(inputs), tf.float16)
        
        # Scaled dot-product attention
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores * self.scale
        
        # Create or apply attention mask - FIX: Ensure dtype consistency
        if attention_mask is None:
            # Create causal mask with proper dtype (float16 to match scores)
            mask = tf.linalg.band_part(tf.ones([seq_len, seq_len], dtype=tf.float16), -1, 0)
            # Use -10000.0 and cast to float16
            mask = tf.where(mask == 0, tf.cast(-10000.0, tf.float16), tf.cast(0.0, tf.float16))
            attention_mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)
        else:
            # Ensure provided mask has correct dtype
            attention_mask = tf.cast(attention_mask, tf.float16)
        
        scores = scores + attention_mask
        
        # Apply softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout_layer(attention_weights, training=training)
        
        # Apply attention to values
        context = tf.matmul(attention_weights, v)
        
        # Transpose and reshape
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, seq_len, self.num_heads * self.key_dim])
        
        # Output projection
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
    
def safe_cast_float16(tensor, name=None):
    """Safely cast tensor to float16 with overflow protection."""
    with tf.name_scope(name or 'safe_cast_float16'):
        # Clip to safe float16 range (leaving some margin)
        tensor = tf.clip_by_value(tensor, -60000.0, 60000.0)
        return tf.cast(tensor, tf.float16)

class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer with improved float16 handling"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_seq_len=2048, use_custom_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_custom_attention = use_custom_attention
        
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections - let mixed precision handle dtype
        self.query_proj = layers.Dense(embed_dim, name='query_proj')
        self.key_proj = layers.Dense(embed_dim, name='key_proj')
        self.value_proj = layers.Dense(embed_dim, name='value_proj')
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
        self.dropout = layers.Dropout(dropout, dtype=tf.float16)

        
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        if self.use_custom_attention:
            # Use custom attention
            attention = self.custom_attention(x, training=training)
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

            q = tf.cast(self.query_proj(x), tf.float16)
            k = tf.cast(self.key_proj(x), tf.float16)
            v = tf.cast(self.value_proj(x), tf.float16)
            
            # Create causal mask - use safer values
            mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            mask = tf.where(mask == 0, -10000.0, 0.0)  # Safer value for float16
            mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)
            
            # Standard attention
            attention = self._scaled_dot_product_attention(q, k, v, mask, training)
            
            # Transpose back and reshape
            attention = tf.transpose(attention, perm=[0, 2, 1, 3])
            attention = tf.reshape(attention, (batch_size, seq_len, self.embed_dim))
            
            # Output projection
            output = self.output_proj(attention)
            
            return output
    
    def _scaled_dot_product_attention(self, q, k, v, mask=None, training=False):
        """Standard scaled dot-product attention with float16 safety"""
        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores += mask
        
        # Apply softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, v)
        
        return output

class FeedForward(layers.Layer):
    """Feed-forward network with improved float16 handling"""
    
    def __init__(self, embed_dim, ffn_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        # Let mixed precision policy handle dtype automatically
        self.dense1 = layers.Dense(ffn_dim, activation='gelu', dtype=tf.float16, name='dense1')
        self.dense2 = layers.Dense(embed_dim, dtype=tf.float16, name='dense2')

        
        # Dropout
        self.dropout = layers.Dropout(dropout, dtype=tf.float16)
 
    def call(self, x, training=False):
        x = tf.cast(x, tf.float16)
        x = self.dense1(x)  # GELU activation
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return tf.cast(x, tf.float16)

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with improved float16 handling."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        use_custom_attention: bool = True,
        use_rotary_embeddings: bool = True,
        name: Optional[str] = "transformer_block",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.use_custom_attention = use_custom_attention
        self.use_rotary_embeddings = use_rotary_embeddings
        
        # Create attention layer
        if use_custom_attention:
            self.attention = CustomMultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                use_rotary_embeddings=use_rotary_embeddings,
                dropout=dropout_rate
            )
        else:
            self.attention = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                max_seq_len=2048,
                use_custom_attention=False
            )
        
        # Feed-forward network
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            ffn_dim=ff_dim,
            dropout=dropout_rate
        )
        
        # Layer normalization - let mixed precision handle dtype
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float16)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float16)
        
        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, dtype=tf.float16)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, dtype=tf.float16)
    
    def call(self, inputs, training=False):
        # Ensure inputs are float16
        inputs = tf.cast(inputs, tf.float16)
        
        # Self-attention and residual connection
        attn_output = self.attention(inputs, training=training)
        attn_output = tf.cast(attn_output, tf.float16)  # Ensure float16
        attn_output = self.dropout1(attn_output, training=training)
        
        # Residual connection with matching dtypes
        out1 = self.ln1(tf.cast(inputs, tf.float16) + tf.cast(attn_output, tf.float16))
        
        # Feed-forward network and residual connection  
        ffn_output = self.ffn(out1, training=training)
        ffn_output = tf.cast(ffn_output, tf.float16)  # Ensure float16
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # Residual connection with matching dtypes
        out2 = self.ln2(tf.cast(out1, tf.float16) + tf.cast(ffn_output, tf.float16))
        
        return out2
    
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
        return config

class EnhancedMiniGPT(Model):
    """Enhanced implementation of MiniGPT with proper float16 handling."""
    
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        
        self.config = config
        
        # Embeddings - let mixed precision handle dtype
        self.token_emb = layers.Embedding(
            config.vocab_size, 
            config.embed_dim,
            dtype=tf.float16,
            name="token_emb"
        )
        
        # Use simple positional embeddings
        self.pos_emb = layers.Embedding(
            config.max_seq_len,
            config.embed_dim,
            dtype=tf.float16,
            name="pos_emb"
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_dim=config.ffn_dim,
                dropout_rate=config.dropout,
                use_custom_attention=config.use_custom_attention,
                use_rotary_embeddings=config.use_rotary_embeddings,
                name=f"transformer_block_{i}"
            )
            for i in range(config.num_layers)
        ]
        
        # Layer normalization
        self.ln_f = layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, 
            dtype=tf.float16,  # Explicit dtype
            name="ln_f"
        )
        
        # Output head (token prediction) - cast to float32 for loss computation
        self.head = layers.Dense(config.vocab_size, name="head", dtype=tf.float32)
    
        # Initialize tokenizer if transformers is available
        self.tokenizer = None
        if HAS_TRANSFORMERS and GPT2TokenizerClass is not None:
            try:
                self.tokenizer = GPT2TokenizerClass.from_pretrained('gpt2')
                if self.tokenizer is not None:
                    logger.info("Successfully initialized GPT2Tokenizer")
                else:
                    logger.warning("Tokenizer initialization returned None")
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer: {str(e)}")
                self.tokenizer = None
        else:
            logger.warning("Transformers library not available, tokenizer not initialized")
    
    def call(self, inputs, training=False, mask=None):
        # Cast inputs to int32 for embedding lookup
        inputs = tf.cast(inputs, tf.int32)
        
        # Get embeddings
        x = self.token_emb(inputs)  # [batch_size, seq_len, embed_dim]
        
        # Positional embeddings
        positions = tf.range(tf.shape(inputs)[1], dtype=tf.int32)
        positions = tf.expand_dims(positions, 0)
        positions = tf.tile(positions, [tf.shape(inputs)[0], 1])
        pos_emb = self.pos_emb(positions)
        
        # CRITICAL: Ensure both embeddings are float16 before addition
        x = tf.cast(x, tf.float16) + tf.cast(pos_emb, tf.float16)
        
        # Transformer blocks - ensure float16 throughout
        for block in self.transformer_blocks:
            x = block(tf.cast(x, tf.float16), training=training)
            x = tf.cast(x, tf.float16)  # Ensure output is float16
        
        # Layer normalization
        x = self.ln_f(tf.cast(x, tf.float16))
        
        # CRITICAL: Cast to float32 for output head
        x_float32 = tf.cast(x, tf.float32)
        logits = self.head(x_float32)
        
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.config.vocab_size,
            'max_seq_len': self.config.max_seq_len,
            'embed_dim': self.config.embed_dim,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'ffn_dim': self.config.ffn_dim,
            'dropout': self.config.dropout,
            'layer_norm_epsilon': self.config.layer_norm_epsilon,
            'use_custom_attention': self.config.use_custom_attention,
            'use_rotary_embeddings': self.config.use_rotary_embeddings
        })
        return config

# Export all necessary classes
__all__ = [
    'ModelConfig',
    'EnhancedMiniGPT',
    'CustomMultiHeadAttention',
    'MultiHeadAttention',
    'RotaryPositionalEmbedding',
    'FeedForward',
    'TransformerBlock',
    'create_dense_float16'
]