import tensorflow as tf
from tensorflow.keras import layers, Model

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
        batch_size: int = 1,  # Changed from 8 to 1 to match example input
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
    """Rotary Positional Embedding layer with float16 support."""
    
    def __init__(self, dim, max_seq_len=1024, dtype=tf.float16, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Ensure all constants are float16
        position = tf.cast(tf.range(max_seq_len), dtype=tf.float16)
        div_term_exp = tf.cast(
            tf.range(0, dim, 2), 
            dtype=tf.float16
        ) * tf.cast(-tf.math.log(10000.0) / dim, dtype=tf.float16)
        
        # Calculate div_term in float16
        div_term = tf.cast(tf.exp(div_term_exp), dtype=tf.float16)
        
        # Ensure position multiplication is in float16
        pos_expanded = tf.expand_dims(position, 1)
        div_term_expanded = tf.expand_dims(div_term, 0)
        pos = tf.cast(
            pos_expanded * div_term_expanded,
            dtype=tf.float16
        )
        
        # Store sin and cos values in float16
        self.sin_vals = tf.cast(tf.sin(pos), dtype=tf.float16)
        self.cos_vals = tf.cast(tf.cos(pos), dtype=tf.float16)
    
    def _apply_rotary_pos_emb(self, x, sin, cos):
        """Apply rotary positional embeddings with float16 support."""
        # Ensure inputs are float16
        x = tf.cast(x, tf.float16)
        sin = tf.cast(sin, tf.float16)
        cos = tf.cast(cos, tf.float16)
        
        x_shape = tf.shape(x)
        seq_len = x_shape[1]
        
        # Ensure broadcasting shapes are float16
        sin = tf.cast(
            tf.expand_dims(sin[:seq_len], 0),
            dtype=tf.float16
        )
        cos = tf.cast(
            tf.expand_dims(cos[:seq_len], 0),
            dtype=tf.float16
        )
        
        # Split and rotate in float16
        x1, x2 = tf.split(x, 2, axis=-1)
        x1 = tf.cast(x1, tf.float16)
        x2 = tf.cast(x2, tf.float16)
        
        rotated_x1 = tf.cast(cos * x1 - sin * x2, dtype=tf.float16)
        rotated_x2 = tf.cast(sin * x1 + cos * x2, dtype=tf.float16)
        
        return tf.concat([rotated_x1, rotated_x2], axis=-1)
    
    def call(self, x, seq_len=None):
        """Apply rotary positional embeddings."""
        x = tf.cast(x, tf.float16)
        
        if seq_len is None:
            seq_len = tf.shape(x)[1]
        
        sin = tf.cast(self.sin_vals[:seq_len], tf.float16)
        cos = tf.cast(self.cos_vals[:seq_len], tf.float16)
        
        return self._apply_rotary_pos_emb(x, sin, cos)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'max_seq_len': self.max_seq_len,
            'dtype': self.dtype
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
        name: Optional[str] = "custom_attention",  # Provide default name
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
        super().__init__(dtype=tf.float16, **kwargs)  # Set layer dtype
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_custom_attention = use_custom_attention
        
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections with float16
        self.query_proj = create_dense_float16(embed_dim, name='query_proj')
        self.key_proj = create_dense_float16(embed_dim, name='key_proj')
        self.value_proj = create_dense_float16(embed_dim, name='value_proj')
        self.output_proj = create_dense_float16(embed_dim, name='output_proj')

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
            
            # Create causal mask and cast to float16
            mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            mask = tf.cast(
                tf.where(mask == 0, -1e9, 0.0),
                dtype=tf.float16
            )
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
        # Cast all inputs to float16
        q = tf.cast(q, tf.float16)
        k = tf.cast(k, tf.float16)
        v = tf.cast(v, tf.float16)

        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float16))
        
        # Apply mask if provided
        if mask is not None:
            mask = tf.cast(mask, tf.float16)
            scores += mask
        
        # Apply softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = tf.cast(attention_weights, tf.float16)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, v)
        
        return output

class FeedForward(layers.Layer):
    """Feed-forward network"""
    
    def __init__(self, embed_dim, ffn_dim, dropout=0.1, **kwargs):
        super().__init__(dtype=tf.float16, **kwargs)
        self.dense1 = create_dense_float16(ffn_dim, activation='gelu', name='dense1')
        self.dense2 = create_dense_float16(embed_dim, name='dense2')
        
        # Dropout
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
        name: Optional[str] = "transformer_block",  # Provide default name
        **kwargs
    ):
        super().__init__(name=name, dtype=tf.float16, **kwargs)
        
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
        
        # Layer normalization
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float16)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float16)
        
        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # Self-attention and residual connection
        attn_output = self.attention(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.ln1(inputs + attn_output)
        
        # Feed-forward network and residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.ln2(out1 + ffn_output)
        
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

class EnhancedMiniGPT(Model):  # Changed from MiniGPT to EnhancedMiniGPT
    """Enhanced implementation of MiniGPT with additional features."""
    
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(dtype=tf.float16, **kwargs)
        
        self.config = config
        
        # Embeddings
        self.token_emb = layers.Embedding(
            config.vocab_size, 
            config.embed_dim,
            dtype=tf.float16,
            name="token_emb"
        )
        self.pos_emb = RotaryPositionalEmbedding(config.embed_dim, config.max_seq_len, name="pos_emb")
        
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
        self.ln_f = layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_f")
        
        # Output head (token prediction)
        self.head = create_dense_float16(config.vocab_size, name="head")
    
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
        # Cast inputs to float16
        inputs = tf.cast(inputs, tf.float16)
        
        # Validate input shape
        input_shape = tf.shape(inputs)
        tf.debugging.assert_rank(inputs, 2, message="Input must be rank 2: [batch_size, seq_len]")
        tf.debugging.assert_less_equal(
            input_shape[1], 
            self.config.max_seq_len,
            message=f"Input sequence length must be <= {self.config.max_seq_len}"
        )
        
        # Get batch size and sequence length
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        
        # Token embeddings
        x = self.token_emb(inputs)  # [batch_size, seq_len, embed_dim]
        
        # Positional embeddings
        x = self.pos_emb(x, seq_len)  # [batch_size, seq_len, embed_dim]
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)  # [batch_size, seq_len, embed_dim]
        
        # Layer normalization
        x = self.ln_f(x)  # [batch_size, seq_len, embed_dim]
        
        # Output logits
        logits = self.head(x)  # [batch_size, seq_len, vocab_size]
        
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
    'TransformerBlock'
]
