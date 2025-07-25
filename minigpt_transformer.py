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
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
#Devansh Sinha 
# Optional: only import if transformers is available
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: transformers library not found. Tokenizer features will be disabled.")
    HAS_TRANSFORMERS = False

#Devansh Sinha
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Devansh Sinha
# NOTE: This model is compatible with mixed precision (fp16) training using tf.keras.mixed_precision.
# Ensure no explicit float32 dtype is set in layers unless required for stability.

#Devansh Sinha
@dataclass
#Devansh Sinha
class MoEConfig:
    """Configuration for the MoE MiniGPT model."""
    vocab_size: int = 10000
    max_seq_len: int = 256
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    use_rotary_embeddings: bool = True
    learning_rate: float = 2e-4
    batch_size: int = 32
    seq_len: int = 256
    #Devansh Sinha
    # MoE specific parameters
    num_experts: int = 4
    top_k_experts: int = 1
    expert_capacity: Optional[int] = None  # Auto-calculated if None
    load_balancing_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    use_moe_layers: List[int] = None  # Which layers to use MoE (None = all)
    #Devansh Sinha
    def __post_init__(self):
        if self.use_moe_layers is None:
            # Use MoE in every other layer starting from layer 2
            self.use_moe_layers = list(range(2, self.num_layers, 2))
    #Devansh Sinha
        if self.expert_capacity is None:
            # Auto-calculate expert capacity
            self.expert_capacity = max(4, (self.seq_len * self.batch_size * self.top_k_experts) // self.num_experts)
#Devansh Sinha
class RotaryPositionalEmbedding(layers.Layer):
    """Rotary Positional Embedding layer with proper weight management."""
    #Devansh Sinha
    def __init__(self, dim=512, max_seq_len=256, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
      #Devansh Sinha  
    def build(self, input_shape):
        super().build(input_shape)
      #Devansh Sinha  
        # Create position indices - ensure int32 dtype
        position = tf.range(self.max_seq_len, dtype=tf.int32)
        position = tf.cast(position, tf.float32)
        div_term = tf.exp(tf.range(0, self.dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.dim))
        #Devansh Sinha
        # Calculate sin and cos values
        pos = tf.expand_dims(position, 1) * tf.expand_dims(div_term, 0)
        #Devansh Sinha
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
        #Devansh Sinha
        # Assign values
        self.sin_vals.assign(tf.sin(pos))
        self.cos_vals.assign(tf.cos(pos))
    #Devansh Sinha
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
        #Devansh Sinha
        return tf.concat([rotated_x1, rotated_x2], axis=-1)
    #Devansh Sinha
    def call(self, x):
        return self._apply_rotary_pos_emb(x, self.sin_vals, self.cos_vals)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'max_seq_len': self.max_seq_len
        })
        return config
#Devansh Sinha
class Expert(layers.Layer):
    """Individual expert network (FFN)."""
    
    def __init__(self, embed_dim=512, ffn_dim=2048, dropout=0.1, expert_id=0, **kwargs):
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
#Devansh Sinha
class Router(layers.Layer):
    """Router network for MoE that decides which experts to use."""
    
    def __init__(self, embed_dim=512, num_experts=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        
        self.router_weights = layers.Dense(
            num_experts,
            use_bias=False,
            kernel_initializer='truncated_normal',
            name='router_weights'
        )
        #Devansh Sinha
    def call(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            router_logits: [batch_size, seq_len, num_experts]
        """
        router_logits = self.router_weights(x)
        return router_logits
#Devansh Sinha
class MixtureOfExperts(layers.Layer):
    """Mixture of Experts layer with efficient routing and load balancing."""
    
    def __init__(self, embed_dim=512, ffn_dim=2048, num_experts=4, top_k=1, 
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
        #Devansh Sinha
        # Router
        self.router = Router(embed_dim, num_experts, name='router')
        
        # Experts
        self.experts = [
            Expert(embed_dim, ffn_dim, dropout, expert_id=i, name=f'expert_{i}')
            for i in range(num_experts)
        ]
        #Devansh Sinha
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
        #Devansh Sinha
        # Load balancing loss using squared coefficient of variation
        mean_usage = tf.reduce_mean(expert_usage)
        variance_usage = tf.reduce_mean(tf.square(expert_usage - mean_usage))
        load_balancing_loss = variance_usage / (mean_usage * mean_usage + 1e-8)
        load_balancing_loss = load_balancing_loss * tf.cast(self.num_experts, tf.float32)
        
        # Router z-loss - prevents router logits from becoming too large
        router_z_loss = tf.reduce_mean(tf.square(tf.reduce_logsumexp(router_logits, axis=-1)))
        
        return load_balancing_loss, router_z_loss
    #Devansh Sinha
    @tf.function
    def call(self, x, training=False):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            output: [batch_size, seq_len, embed_dim]
            aux_losses: dict with auxiliary losses
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
        #Devansh Sinha
        # Process each position with its top-k experts using tf.function compatible approach
        for k in range(self.top_k):
            # Get the k-th expert index and weight for each position
            expert_indices = top_k_indices[:, :, k]  # [batch_size, seq_len]
            expert_weights = top_k_probs[:, :, k]    # [batch_size, seq_len]
            
            # Process each expert using tf.cond for conditional execution
            for expert_idx in range(self.num_experts):
                # Create mask for positions that route to this expert for this k
                expert_mask = tf.equal(expert_indices, expert_idx)  # [batch_size, seq_len]
                
                # Use tf.cond to conditionally apply expert
                def apply_expert():
                    expert_output = self.experts[expert_idx](x, training=training)
                    expert_mask_expanded = tf.expand_dims(tf.cast(expert_mask, tf.float32), -1)
                    weights_expanded = tf.expand_dims(expert_weights, -1)
                    return expert_output * expert_mask_expanded * weights_expanded
                
                def skip_expert():
                    return tf.zeros_like(x)
                
                # Check if any element in the mask is True
                has_tokens = tf.reduce_any(expert_mask)
                expert_contribution = tf.cond(has_tokens, apply_expert, skip_expert)
                output += expert_contribution
        #Devansh Sinha
        # Return output and auxiliary losses as a tuple
        aux_losses = {
            'load_balancing_loss': load_balancing_loss * self.load_balancing_loss_weight,
            'router_z_loss': router_z_loss * self.router_z_loss_weight
        }
        
        return output, aux_losses

class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer with optional rotary embeddings"""
    
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, max_seq_len=256, 
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
        #Devansh Sinha
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
        #Devansh Sinha
        # Create causal mask - ensure int32 dtype
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)
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
    
    def __init__(self, embed_dim=512, ffn_dim=2048, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        #Devansh Sinha
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
    
    def __init__(self, embed_dim=512, num_heads=8, ffn_dim=2048, dropout=0.1, 
                 layer_norm_epsilon=1e-5, max_seq_len=256, block_id=0, 
                 use_moe=False, num_experts=4, top_k_experts=1, 
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
        #Devansh Sinha
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
        #Devansh Sinha
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
        
        if self.use_moe:
            ffn_output, aux_losses = self.ffn(ffn_input, training=training)
            # Return auxiliary losses along with the output
            ffn_output = self.dropout2(ffn_output, training=training)
            x = x + ffn_output
            return x, aux_losses
        else:
            ffn_output = self.ffn(ffn_input, training=training)
            ffn_output = self.dropout2(ffn_output, training=training)
            x = x + ffn_output
            return x, {}
#Devansh Sinha
class MoEMiniGPT(Model):
    """MiniGPT model with Mixture of Experts."""
    
    def __init__(self, config: MoEConfig = None, tokenizer_path="my-10k-bpe-tokenizer", **kwargs):
        super().__init__(**kwargs)
        self.config = config or MoEConfig()
        self._tokenizer = None

        # Load custom ByteLevelBPETokenizer
        try:
            tokenizer = ByteLevelBPETokenizer(
                vocab=f"{tokenizer_path}/vocab.json",
                merges=f"{tokenizer_path}/merges.txt",
            )
            self._tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="<unk>",
                pad_token="<pad>",
                bos_token="<s>",
                eos_token="</s>",
                mask_token="<mask>",
            )
        except Exception as e:
            print(f"Failed to load custom tokenizer: {e}")
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
        #Devansh Sinha
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
        #Devansh Sinha
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
    #Devansh Sinha
    def call(self, inputs, mask=None, training=False):
        """Fixed forward pass that properly handles auxiliary losses."""
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            mask = inputs.get('attention_mask', mask)
        else:
            input_ids = inputs
        
        # Ensure input_ids is int32
        input_ids = tf.cast(input_ids, tf.int32)
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
        
        # Collect auxiliary losses
        total_aux_losses = {}
        #Devansh Sinha
        # Apply transformer blocks
        for block in self.transformer_blocks:
            if hasattr(block, 'use_moe') and block.use_moe:
                x, aux_losses = block(x, mask=mask, training=training)
                # Accumulate auxiliary losses
                for loss_name, loss_value in aux_losses.items():
                    if loss_name in total_aux_losses:
                        total_aux_losses[loss_name] += loss_value
                    else:
                        total_aux_losses[loss_name] = loss_value
            else:
                x, _ = block(x, mask=mask, training=training)
        
        # Final layer norm
        x = self.final_layer_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        return logits, total_aux_losses

    def compute_loss_method(self, input_ids, labels=None):
        """Fixed compute loss method that properly handles auxiliary losses."""
        # Ensure input_ids is int32
        input_ids = tf.cast(input_ids, tf.int32)
        
        # Ensure input_ids is properly shaped
        if len(input_ids.shape) == 1:
            input_ids = tf.expand_dims(input_ids, 0)
        #Devansh Sinha
        # Handle the case where labels are None - create labels from input_ids
        if labels is None:
            # For language modeling, shift input_ids to create labels
            input_for_model = input_ids[:, :-1]  # All tokens except last
            labels = input_ids[:, 1:]            # All tokens except first (shifted)
        else:
            # Labels provided explicitly - ensure proper shape and dtype
            labels = tf.cast(labels, tf.int32)
            if len(labels.shape) == 1:
                labels = tf.expand_dims(labels, 0)
            input_for_model = input_ids
        
        # Validate that we have valid labels
        if labels is None:
            raise ValueError("Labels cannot be None after processing")
        
        # Get model predictions
        logits, aux_losses = self(input_for_model, training=True)
        
        # Ensure shapes are compatible
        batch_size = tf.shape(labels)[0]
        seq_len = tf.shape(labels)[1]
        vocab_size = tf.shape(logits)[-1]
        
        # Reshape for loss computation
        labels_flat = tf.reshape(labels, [-1])  # [batch_size * seq_len]
        logits_flat = tf.reshape(logits, [-1, vocab_size])  # [batch_size * seq_len, vocab_size]
        #Devansh Sinha
        # Create mask to ignore invalid tokens (assuming -100 is used for padding/invalid tokens)
        pad_token_id = -100
        valid_mask = tf.not_equal(labels_flat, pad_token_id)
        
        # Compute sparse categorical crossentropy loss
        losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_flat, 
            logits_flat, 
            from_logits=True
        )
        #Devansh Sinha
        # Apply mask to ignore padded positions
        num_valid = tf.reduce_sum(tf.cast(valid_mask, tf.float32))
        if tf.greater(num_valid, 0):
            # Only compute loss on valid tokens
            masked_losses = tf.where(valid_mask, losses, 0.0)
            main_loss = tf.reduce_sum(masked_losses) / tf.maximum(num_valid, 1.0)
        else:
            # If no valid mask, compute mean loss
            main_loss = tf.reduce_mean(losses)
        
        # Add auxiliary losses from MoE layers
        auxiliary_loss = 0.0
        if aux_losses:
            auxiliary_loss = tf.add_n([v for v in aux_losses.values()])
        total_loss = main_loss + auxiliary_loss
        
        return total_loss

    def get_expert_utilization(self):
        """Get expert utilization statistics from MoE layers."""
        utilization_stats = {}
        
        for i, block in enumerate(self.transformer_blocks):
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'router'):
                layer_name = f"layer_{i}"
                utilization_stats[layer_name] = {
                    'is_moe_layer': True,
                    'num_experts': block.ffn.num_experts,
                    'top_k': block.ffn.top_k
                }
            else:
                layer_name = f"layer_{i}"
                utilization_stats[layer_name] = {
                    'is_moe_layer': False
                }
        #Devansh Sinha
        return utilization_stats
    
    def get_model_info(self):
        """Get comprehensive model information."""
        # Ensure model is built
        if not self.built:
            dummy_input = tf.ones((1, self.config.max_seq_len), dtype=tf.int32)
            _ = self(dummy_input)
        total_params = sum([tf.reduce_prod(var.shape) for var in self.trainable_variables])
        
        moe_layers = [i for i in range(self.config.num_layers) if i in self.config.use_moe_layers]
        standard_layers = [i for i in range(self.config.num_layers) if i not in self.config.use_moe_layers]
        #Devansh Sinha
        info = {
            'model_type': 'MoE MiniGPT',
            'total_parameters': int(total_params),
            'vocab_size': self.config.vocab_size,
            'embed_dim': self.config.embed_dim,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'max_seq_len': self.config.max_seq_len,
            'moe_config': {
                'num_experts': self.config.num_experts,
                'top_k_experts': self.config.top_k_experts,
                'moe_layers': moe_layers,
                'standard_layers': standard_layers,
                'expert_capacity': self.config.expert_capacity
            },
            'tokenizer_available': self.tokenizer is not None
        }
        
        return info
    
    def save_model(self, filepath: str):
        """Save model weights and configuration."""
        # Save model weights
        self.save_weights(filepath + '_weights')
        
        # Save configuration
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'max_seq_len': self.config.max_seq_len,
            'embed_dim': self.config.embed_dim,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'ffn_dim': self.config.ffn_dim,
            'dropout': self.config.dropout,
            'layer_norm_epsilon': self.config.layer_norm_epsilon,
            'use_rotary_embeddings': self.config.use_rotary_embeddings,
            'num_experts': self.config.num_experts,
            'top_k_experts': self.config.top_k_experts,
            'expert_capacity': self.config.expert_capacity,
            'load_balancing_loss_weight': self.config.load_balancing_loss_weight,
            'router_z_loss_weight': self.config.router_z_loss_weight,
            'use_moe_layers': self.config.use_moe_layers
        }
        #Devansh Sinha
        with open(filepath + '_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    #Devansh Sinha
    @classmethod
    def load_model(cls, filepath: str):
        """Load model from saved weights and configuration."""
        # Load configuration
        with open(filepath + '_config.json', 'r') as f:
            config_dict = json.load(f)
        
        config = MoEConfig(**config_dict)
        model = cls(config)
        
        # Build model by calling it with dummy input
        dummy_input = tf.ones((1, config.max_seq_len), dtype=tf.int32)
        _ = model(dummy_input)
        
        # Load weights
        model.load_weights(filepath + '_weights')
        
        logger.info(f"Model loaded from {filepath}")
        return model

    def generate_text(self, prompt, max_length=50, temperature=1.0):
        """
        Simple greedy text generation for demonstration.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not available for text generation.")
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="tf", max_length=self.config.max_seq_len, truncation=True
        )
        for _ in range(max_length):
            logits, _ = self(input_ids)
            logits = logits[:, -1, :] / temperature
            next_token_id = tf.argmax(logits, axis=-1, output_type=tf.int32)
            next_token_id = tf.expand_dims(next_token_id, axis=-1)
            input_ids = tf.concat([input_ids, next_token_id], axis=-1)
            # Stop if EOS token is generated
            if hasattr(self.tokenizer, "eos_token_id") and next_token_id[0, 0].numpy() == self.tokenizer.eos_token_id:
                break
        output_ids = input_ids.numpy()[0]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

# Example usage
def create_sample_model():
    """Create a sample MoE MiniGPT model for testing."""
    config = MoEConfig(
        vocab_size=10000,  # Set vocab size to 10000
        max_seq_len=256,
        embed_dim=512,
        num_heads=8,
        num_layers=8,
        ffn_dim=2048,
        num_experts=4,
        top_k_experts=1,
        use_moe_layers=[2, 4, 6],
        batch_size=32,
        seq_len=256
    )
    model = MoEMiniGPT(config, tokenizer_path="my-10k-bpe-tokenizer")
    dummy_input = tf.ones((1, config.max_seq_len), dtype=tf.int32)
    _ = model(dummy_input)
    logger.info("Sample MoE MiniGPT model created successfully!")
    logger.info(f"Model info: {model.get_model_info()}")
    return model
#Devansh Sinha

def create_dummy_dataset(vocab_size: int = 10000, seq_len: int = 256, 
                        batch_size: int = 48, num_batches: int = 100):
    """Create a dummy dataset for testing, yielding (input_ids, labels) pairs."""
    def generator():
        for _ in range(num_batches):
            # Generate random token sequences
            batch = tf.random.uniform(
                (batch_size, seq_len), 
                minval=0, 
                maxval=vocab_size, 
                dtype=tf.int32
            )
            # For language modeling, labels are input_ids shifted by one
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            yield input_ids, labels
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, seq_len-1), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, seq_len-1), dtype=tf.int32)
        )
    )
    return dataset

#Devansh Sinha
if __name__ == "__main__":
    # Example usage
    model = create_sample_model()

    # Interactive chat loop
    if model.tokenizer is not None and hasattr(model, "generate_text"):
        print("MiniGPT Chat! Type 'quit' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() == "quit":
                print("Exiting chat.")
                break
            response = model.generate_text(user_input, max_length=50)
            print(f"MiniGPT: {response}")
    else:
        print("Tokenizer not available or generate_text not implemented. Model created successfully but text generation requires transformers library.")

    # Display model information
    print(f"Model info: {model.get_model_info()}")
    print(f"Expert utilization: {model.get_expert_utilization()}")

