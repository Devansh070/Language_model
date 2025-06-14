import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import os
import logging
import time
import math
import json

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
class MoEConfig:
    """Configuration for the MoE MiniGPT model."""
    vocab_size: int = 50257
    max_seq_len: int = 1024
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    use_rotary_embeddings: bool = True
    learning_rate: float = 1e-4
    batch_size: int = 1
    seq_len: int = 1024
    
    # MoE specific parameters
    num_experts: int = 8
    top_k_experts: int = 2
    expert_capacity: Optional[int] = None  # Auto-calculated if None
    load_balancing_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    use_moe_layers: List[int] = None  # Which layers to use MoE (None = all)
    
    def __post_init__(self):
        if self.use_moe_layers is None:
            # Use MoE in every other layer starting from layer 2
            self.use_moe_layers = list(range(2, self.num_layers, 2))
        
        if self.expert_capacity is None:
            # Auto-calculate expert capacity
            self.expert_capacity = max(4, (self.seq_len * self.batch_size * self.top_k_experts) // self.num_experts)

class RotaryPositionalEmbedding(layers.Layer):
    """Rotary Positional Embedding layer with proper weight management."""
    
    def __init__(self, dim=768, max_seq_len=1024, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        
    def build(self, input_shape):
        super().build(input_shape)
        
        # Create position indices
        position = tf.range(self.max_seq_len, dtype=tf.float32)
        div_term = tf.exp(tf.range(0, self.dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.dim))
        
        # Calculate sin and cos values
        pos = tf.expand_dims(position, 1) * tf.expand_dims(div_term, 0)
        
        # Store as non-trainable weights
        self.sin_vals = self.add_weight(
            name='sin_vals',
            shape=(self.max_seq_len, self.dim // 2),
            initializer='zeros',
            trainable=False
        )
        self.cos_vals = self.add_weight(
            name='cos_vals',
            shape=(self.max_seq_len, self.dim // 2),
            initializer='zeros',
            trainable=False
        )
        
        # Assign values
        self.sin_vals.assign(tf.sin(pos))
        self.cos_vals.assign(tf.cos(pos))
    
    def _apply_rotary_pos_emb(self, x, sin, cos):
        """Apply rotary positional embeddings to input tensor."""
        seq_len = tf.shape(x)[2]
        sin = sin[:seq_len]
        cos = cos[:seq_len]
        sin = tf.expand_dims(tf.expand_dims(sin, 0), 0)
        cos = tf.expand_dims(tf.expand_dims(cos, 0), 0)
        
        x1, x2 = tf.split(x, 2, axis=-1)
        rotated_x1 = cos * x1 - sin * x2
        rotated_x2 = sin * x1 + cos * x2
        
        return tf.concat([rotated_x1, rotated_x2], axis=-1)
    
    def call(self, x):
        return self._apply_rotary_pos_emb(x, self.sin_vals, self.cos_vals)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'max_seq_len': self.max_seq_len
        })
        return config

class Expert(layers.Layer):
    """Individual expert network (FFN)."""
    
    def __init__(self, embed_dim=768, ffn_dim=3072, dropout=0.1, expert_id=0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.expert_id = expert_id
        
        self.dense1 = layers.Dense(ffn_dim, activation='gelu', name=f'expert_{expert_id}_dense1')
        self.dense2 = layers.Dense(embed_dim, name=f'expert_{expert_id}_dense2')
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

class Router(layers.Layer):
    """Router network for MoE that decides which experts to use."""
    
    def __init__(self, embed_dim=768, num_experts=8, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        
        self.router_weights = layers.Dense(
            num_experts,
            use_bias=False,
            kernel_initializer='truncated_normal',
            name='router_weights'
        )
        
    def call(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            router_logits: [batch_size, seq_len, num_experts]
        """
        router_logits = self.router_weights(x)
        return router_logits

class MixtureOfExperts(layers.Layer):
    """Mixture of Experts layer with efficient routing and load balancing."""
    
    def __init__(self, embed_dim=768, ffn_dim=3072, num_experts=8, top_k=2, 
                 expert_capacity=None, dropout=0.1, load_balancing_loss_weight=0.01,
                 router_z_loss_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity or 32
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.router_z_loss_weight = router_z_loss_weight
        
        # Router
        self.router = Router(embed_dim, num_experts, name='router')
        
        # Experts
        self.experts = [
            Expert(embed_dim, ffn_dim, dropout, expert_id=i, name=f'expert_{i}')
            for i in range(num_experts)
        ]
        
    def _compute_routing_probabilities(self, router_logits):
        """Compute routing probabilities and auxiliary losses."""
        # Apply softmax to get probabilities
        router_probs = tf.nn.softmax(router_logits, axis=-1)
        
        # Get top-k routing decisions
        top_k_probs, top_k_indices = tf.nn.top_k(router_probs, k=self.top_k)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / (tf.reduce_sum(top_k_probs, axis=-1, keepdims=True) + 1e-8)
        
        return router_probs, top_k_probs, top_k_indices
    
    def _compute_auxiliary_losses(self, router_logits, router_probs):
        """Compute load balancing and router z-loss."""
        # Load balancing loss - encourages equal utilization of experts
        # Average probability of routing to each expert
        expert_usage = tf.reduce_mean(router_probs, axis=[0, 1])  # [num_experts]
        
        # Load balancing loss using squared coefficient of variation
        mean_usage = tf.reduce_mean(expert_usage)
        variance_usage = tf.reduce_mean(tf.square(expert_usage - mean_usage))
        load_balancing_loss = variance_usage / (mean_usage * mean_usage + 1e-8)
        load_balancing_loss = load_balancing_loss * tf.cast(self.num_experts, tf.float32)
        
        # Router z-loss - prevents router logits from becoming too large
        router_z_loss = tf.reduce_mean(tf.square(tf.reduce_logsumexp(router_logits, axis=-1)))
        
        return load_balancing_loss, router_z_loss
    
    def call(self, x, training=False):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Get router decisions
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        router_probs, top_k_probs, top_k_indices = self._compute_routing_probabilities(router_logits)
        
        # Compute auxiliary losses
        load_balancing_loss, router_z_loss = self._compute_auxiliary_losses(router_logits, router_probs)
        
        # Initialize output
        output = tf.zeros_like(x)
        
        # Process each position with its top-k experts
        for k in range(self.top_k):
            # Get the k-th expert index and weight for each position
            expert_indices = top_k_indices[:, :, k]  # [batch_size, seq_len]
            expert_weights = top_k_probs[:, :, k]    # [batch_size, seq_len]
            
            # Process each expert
            for expert_idx in range(self.num_experts):
                # Create mask for positions that route to this expert for this k
                expert_mask = tf.equal(expert_indices, expert_idx)  # [batch_size, seq_len]
                
                if tf.reduce_any(expert_mask):
                    # Apply expert to all positions (we'll mask the output)
                    expert_output = self.experts[expert_idx](x, training=training)  # [batch_size, seq_len, embed_dim]
                    
                    # Apply mask and weights
                    expert_mask_expanded = tf.expand_dims(tf.cast(expert_mask, tf.float32), -1)  # [batch_size, seq_len, 1]
                    weights_expanded = tf.expand_dims(expert_weights, -1)  # [batch_size, seq_len, 1]
                    
                    # Add weighted contribution to output
                    output += expert_output * expert_mask_expanded * weights_expanded
        
        # Add auxiliary losses
        self.add_loss(load_balancing_loss * self.load_balancing_loss_weight)
        self.add_loss(router_z_loss * self.router_z_loss_weight)
        
        return output

class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer with optional rotary embeddings"""
    
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1, max_seq_len=1024, 
                 use_rotary_embeddings=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_rotary_embeddings = use_rotary_embeddings
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.query_proj = layers.Dense(embed_dim, use_bias=False, name='query_proj')
        self.key_proj = layers.Dense(embed_dim, use_bias=False, name='key_proj')
        self.value_proj = layers.Dense(embed_dim, use_bias=False, name='value_proj')
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name='output_proj')
        
        if use_rotary_embeddings:
            self.rope = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_seq_len=max_seq_len,
                name='rope'
            )
        
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x, mask=None, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        
        q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
        if self.use_rotary_embeddings:
            q = self.rope(q)
            k = self.rope(k)
        
        scores = tf.matmul(q, k, transpose_b=True) * self.scale
        
        # Create causal mask
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.where(tf.equal(causal_mask, 0), -1e9, 0.0)
        causal_mask = tf.expand_dims(tf.expand_dims(causal_mask, 0), 0)
        
        scores = scores + causal_mask
        
        if mask is not None:
            mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
            mask = tf.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask) * -1e9
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        attention_output = tf.matmul(attention_weights, v)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.embed_dim))
        
        output = self.output_proj(attention_output)
        return output

class FeedForward(layers.Layer):
    """Standard Feed-forward network (used in non-MoE layers)"""
    
    def __init__(self, embed_dim=768, ffn_dim=3072, dropout=0.1, **kwargs):
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

class MoETransformerBlock(layers.Layer):
    """Transformer block with optional MoE FFN"""
    
    def __init__(self, embed_dim=768, num_heads=12, ffn_dim=3072, dropout=0.1, 
                 layer_norm_epsilon=1e-5, max_seq_len=1024, block_id=0, 
                 use_moe=False, num_experts=8, top_k_experts=2, 
                 load_balancing_loss_weight=0.01, router_z_loss_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        
        self.use_moe = use_moe
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_rotary_embeddings=True,
            name=f'attention_{block_id}'
        )
        
        # Feed-forward network (MoE or standard)
        if use_moe:
            self.ffn = MixtureOfExperts(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                num_experts=num_experts,
                top_k=top_k_experts,
                dropout=dropout,
                load_balancing_loss_weight=load_balancing_loss_weight,
                router_z_loss_weight=router_z_loss_weight,
                name=f'moe_ffn_{block_id}'
            )
        else:
            self.ffn = FeedForward(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                name=f'ffn_{block_id}'
            )
        
        # Layer normalization
        self.ln1 = layers.LayerNormalization(epsilon=layer_norm_epsilon, name=f'ln1_{block_id}')
        self.ln2 = layers.LayerNormalization(epsilon=layer_norm_epsilon, name=f'ln2_{block_id}')
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout, name=f'dropout1_{block_id}')
        self.dropout2 = layers.Dropout(dropout, name=f'dropout2_{block_id}')
        
    def call(self, x, mask=None, training=False):
        # Pre-norm attention with residual connection
        attn_input = self.ln1(x)
        attn_output = self.attention(attn_input, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output
        
        # Pre-norm feed-forward with residual connection
        ffn_input = self.ln2(x)
        ffn_output = self.ffn(ffn_input, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = x + ffn_output
        
        return x

class MoEMiniGPT(Model):
    """MiniGPT model with Mixture of Experts."""
    
    def __init__(self, config: MoEConfig = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or MoEConfig()
        self._tokenizer = None
        
        # Initialize tokenizer if available
        if HAS_TRANSFORMERS:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                logger.info("Successfully loaded GPT-2 tokenizer")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                self._tokenizer = None
        
        # Build model layers
        self._build_layers()
        
    def _build_layers(self):
        """Build the model architecture."""
        # Token embedding layer
        self.token_embedding = layers.Embedding(
            self.config.vocab_size,
            self.config.embed_dim,
            name="token_embedding"
        )
        
        # Positional embedding layer (only if not using rotary)
        if not self.config.use_rotary_embeddings:
            self.pos_embedding = layers.Embedding(
                self.config.max_seq_len,
                self.config.embed_dim,
                name="position_embedding"
            )
        
        # Transformer blocks (some with MoE, some without)
        self.transformer_blocks = []
        for i in range(self.config.num_layers):
            use_moe = i in self.config.use_moe_layers
            block = MoETransformerBlock(
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                ffn_dim=self.config.ffn_dim,
                dropout=self.config.dropout,
                layer_norm_epsilon=self.config.layer_norm_epsilon,
                max_seq_len=self.config.max_seq_len,
                block_id=i,
                use_moe=use_moe,
                num_experts=self.config.num_experts,
                top_k_experts=self.config.top_k_experts,
                load_balancing_loss_weight=self.config.load_balancing_loss_weight,
                router_z_loss_weight=self.config.router_z_loss_weight,
                name=f"transformer_block_{i}"
            )
            self.transformer_blocks.append(block)
        
        # Final layer norm
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=self.config.layer_norm_epsilon,
            name="final_layer_norm"
        )
        
        # Output projection
        self.output_projection = layers.Dense(
            self.config.vocab_size,
            use_bias=False,
            name="output_projection"
        )
        
        # Dropout
        self.dropout = layers.Dropout(self.config.dropout, name="embedding_dropout")
        
    @property
    def tokenizer(self):
        return self._tokenizer
    
    def call(self, inputs, mask=None, training=False):
        """Forward pass of the model."""
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            mask = inputs.get('attention_mask', mask)
        else:
            input_ids = inputs
            
        seq_len = tf.shape(input_ids)[1]
        
        # Get token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional embeddings if not using rotary
        if not self.config.use_rotary_embeddings:
            positions = tf.range(seq_len, dtype=tf.int32)
            position_embeddings = self.pos_embedding(positions)
            x = x + position_embeddings
        
        # Apply dropout to embeddings
        x = self.dropout(x, training=training)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=mask, training=training)
        
        # Final layer norm
        x = self.final_layer_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
        
    def generate(self, input_ids, max_length: int = 100, temperature: float = 0.7,
                top_k: int = 50, top_p: float = 0.9, **kwargs):
        """Generate text using the model with proper sampling."""
        if len(input_ids.shape) == 1:
            input_ids = tf.expand_dims(input_ids, 0)
        
        batch_size = tf.shape(input_ids)[0]
        output_sequence = input_ids
        
        for step in range(max_length):
            if tf.shape(output_sequence)[1] >= self.config.max_seq_len:
                output_sequence = output_sequence[:, -self.config.max_seq_len:]
                
            logits = self(output_sequence, training=False)
            next_token_logits = logits[:, -1, :]
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            if top_k > 0:
                top_k_logits, _ = tf.nn.top_k(next_token_logits, k=top_k)
                next_token_logits = tf.where(
                    next_token_logits < top_k_logits[:, -1:],
                    tf.ones_like(next_token_logits) * -1e9,
                    next_token_logits
                )
            
            probs = tf.nn.softmax(next_token_logits, axis=-1)
            next_token = tf.random.categorical(tf.math.log(probs + 1e-10), num_samples=1)
            
            output_sequence = tf.concat([output_sequence, next_token], axis=1)
            
            if self.tokenizer is not None:
                eos_token_id = self.tokenizer.eos_token_id
                if tf.reduce_any(tf.equal(next_token, eos_token_id)):
                    break
        
        return output_sequence
        
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7):
        """Generate text from a string prompt (requires tokenizer)"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available. Please install transformers package.")
            
        input_ids = self.tokenizer.encode(prompt, return_tensors='tf')
        output_ids = self.generate(input_ids, max_length=max_length, temperature=temperature)
        output_text = self.tokenizer.decode(output_ids[0].numpy(), skip_special_tokens=True)
        
        return output_text
        
    def compute_loss(self, input_ids, labels=None):
        """Compute the loss for training including MoE auxiliary losses."""
        # Ensure input_ids is properly shaped
        if len(input_ids.shape) == 1:
            input_ids = tf.expand_dims(input_ids, 0)
        
        # Handle the case where labels are None - create labels from input_ids
        if labels is None:
            # For language modeling, shift input_ids to create labels
            input_for_model = input_ids[:, :-1]  # All tokens except last
            labels = input_ids[:, 1:]            # All tokens except first (shifted)
        else:
            # Labels provided explicitly - ensure proper shape
            if len(labels.shape) == 1:
                labels = tf.expand_dims(labels, 0)
            input_for_model = input_ids
        
        # Validate that we have valid labels
        if labels is None:
            raise ValueError("Labels cannot be None after processing")
        
        # Get model predictions
        logits = self(input_for_model, training=True)
        
        # Ensure shapes are compatible
        batch_size = tf.shape(labels)[0]
        seq_len = tf.shape(labels)[1]
        vocab_size = tf.shape(logits)[-1]
        
        # Reshape for loss computation
        labels_flat = tf.reshape(labels, [-1])  # [batch_size * seq_len]
        logits_flat = tf.reshape(logits, [-1, vocab_size])  # [batch_size * seq_len, vocab_size]
        
        # Create mask to ignore invalid tokens (assuming -100 is used for padding/invalid tokens)
        pad_token_id = -100
        valid_mask = tf.not_equal(labels_flat, pad_token_id)
        
        # Compute sparse categorical crossentropy loss
        losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_flat, 
            logits_flat, 
            from_logits=True
        )
        
        # Apply mask to ignore padded positions
        if tf.reduce_any(valid_mask):
            # Only compute loss on valid tokens
            masked_losses = tf.where(valid_mask, losses, 0.0)
            main_loss = tf.reduce_sum(masked_losses) / tf.maximum(tf.reduce_sum(tf.cast(valid_mask, tf.float32)), 1.0)
        else:
            # If no valid mask, compute mean loss
            main_loss = tf.reduce_mean(losses)
        
        # Add auxiliary losses from MoE layers
        auxiliary_losses = tf.reduce_sum(self.losses) if self.losses else 0.0
        total_loss = main_loss + auxiliary_losses
        
        return total_loss

    def get_expert_utilization_stats(self):
        """Get statistics about expert utilization across MoE layers."""
        stats = {}
        for i, block in enumerate(self.transformer_blocks):
            if hasattr(block, 'use_moe') and block.use_moe:
                stats[f'layer_{i}'] = {
                    'num_experts': self.config.num_experts,
                    'top_k': self.config.top_k_experts,
                    'layer_type': 'MoE'
                }
            else:
                stats[f'layer_{i}'] = {
                    'layer_type': 'Standard FFN'
                }
        return stats

# Utility functions
def create_moe_model(config: MoEConfig = None) -> MoEMiniGPT:
    """Create and return a MoE MiniGPT model with the given configuration."""
    if config is None:
        config = MoEConfig()
    
    model = MoEMiniGPT(config)
    
    # Build model by calling it with dummy input
    dummy_input = tf.random.uniform((1, config.seq_len), maxval=config.vocab_size, dtype=tf.int32)
    _ = model(dummy_input)
    
    logger.info(f"Created MoE MiniGPT model with {model.count_params():,} parameters")
    
    # Log model configuration
    logger.info("Model configuration:")
    logger.info(f"  Vocab size: {config.vocab_size}")
    logger.info(f"  Sequence length: {config.seq_len}")
    logger.info(f"  Embedding dimension: {config.embed_dim}")
    logger.info(f"  Number of heads: {config.num_heads}")
    logger.info(f"  Number of layers: {config.num_layers}")
    logger.info(f"  Number of experts: {config.num_experts}")
    logger.info(f"  Top-k experts: {config.top_k_experts}")
    logger.info(f"  MoE layers: {config.use_moe_layers}")
    
    # Log expert utilization statistics
    expert_stats = model.get_expert_utilization_stats()
    logger.info("\nExpert utilization by layer:")
    for layer, stats in expert_stats.items():
        logger.info(f"  {layer}: {stats}")
    
    return model

if __name__ == "__main__":
    # Create and test model
    logger.info("Testing MoE MiniGPT model creation...")
    
    try:
        # Create configuration
        config = MoEConfig(
            vocab_size=50257,
            max_seq_len=1024,
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            ffn_dim=3072,
            dropout=0.1,
            num_experts=8,
            top_k_experts=2,
            use_moe_layers=[2, 4, 6, 8, 10]
        )
        
        # Create model
        model = create_moe_model(config)
        
        # Test generation if tokenizer is available
        if model.tokenizer is not None:
            test_prompt = "The future of artificial intelligence"
            logger.info(f"\nTesting generation with prompt: '{test_prompt}'")
            
            generated_text = model.generate_text(
                prompt=test_prompt,
                max_length=50,
                temperature=0.7
            )
            logger.info(f"Generated text: {generated_text}")
        
        logger.info("Model testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model testing: {e}")
        raise