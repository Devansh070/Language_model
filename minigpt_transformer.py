import tensorflow as tf
import numpy as np
import sentencepiece as sptokenizer
import json
import os
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

keras = tf.keras
layers = tf.keras.layers

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

@dataclass
class ModelConfig:
    """Configuration class for MiniGPT model."""
    vocab_size: int = 25000
    max_seq_len: int = 256
    embed_dim: int = 512
    num_heads: int = 4
    num_layers: int = 12
    ffn_dim: int = 1024
    num_experts: int = 4
    dropout: float = 0.1
    block_size: int = 128
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True

class FlashAttention(layers.Layer):
    """Memory-efficient Flash Attention implementation."""
    
    def __init__(self, embed_dim, num_heads, block_size=64, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.scale = 1.0 / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = layers.Dense(embed_dim, use_bias=False)
        self.k_proj = layers.Dense(embed_dim, use_bias=False)
        self.v_proj = layers.Dense(embed_dim, use_bias=False)
        self.out_proj = layers.Dense(embed_dim, use_bias=False)

    def _split_heads(self, x, batch_size, seq_len):
        """Split the last dimension into (num_heads, head_dim)."""
        x = tf.reshape(x, [batch_size, seq_len, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])  # [batch, heads, seq, head_dim]

    def _flash_attention_block(self, q_block, k, v, causal_mask=None):
        """Compute attention for a block of queries."""
        # Compute attention scores
        scores = tf.matmul(q_block, k, transpose_b=True) * self.scale
        
        # Apply causal mask if provided
        if causal_mask is not None:
            scores = tf.where(causal_mask, scores, -1e9)
        
        # Compute attention weights and output
        weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(weights, v)
        
        return output, weights

    def call(self, x, attention_mask=None, training=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Project to q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Split heads
        q = self._split_heads(q, batch_size, seq_len)
        k = self._split_heads(k, batch_size, seq_len)
        v = self._split_heads(v, batch_size, seq_len)
        
        # Create causal mask
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.cast(causal_mask, tf.bool)
        causal_mask = tf.reshape(causal_mask, (1, 1, seq_len, seq_len))
        
        # Flash attention computation in blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        outputs = []
        
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = tf.minimum(start_idx + self.block_size, seq_len)
            
            q_block = q[:, :, start_idx:end_idx, :]
            mask_block = causal_mask[:, :, start_idx:end_idx, :]
            
            block_output, _ = self._flash_attention_block(q_block, k, v, mask_block)
            outputs.append(block_output)
        
        # Concatenate block outputs
        attn_output = tf.concat(outputs, axis=2)
        
        # Reshape and project
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [batch_size, seq_len, self.embed_dim])
        
        return self.out_proj(attn_output)

class RotaryEmbedding(layers.Layer):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim=None, max_seq_len=2048, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len

    def build(self, input_shape):
        if self.dim is None:
            self.dim = int(input_shape[-1])
        assert self.dim % 2 == 0, "RotaryEmbedding: dim must be even"
        
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.dim // 2, dtype=np.float32) / (self.dim // 2)))
        t = np.arange(self.max_seq_len, dtype=np.float32)
        freqs = np.einsum('i,j->ij', t, inv_freq)
        emb = np.concatenate([np.sin(freqs), np.cos(freqs)], axis=-1)
        
        self.rotary_emb = self.add_weight(
            name="rotary_emb",
            shape=emb.shape,
            initializer=tf.constant_initializer(emb),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        dim = tf.shape(x)[-1]
        rotary = self.rotary_emb[:seq_len, :dim]
        rotary = tf.expand_dims(rotary, 0)
        
        x1, x2 = tf.split(x, 2, axis=-1)
        sin, cos = tf.split(rotary, 2, axis=-1)
        
        return tf.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

class MixtureOfExperts(layers.Layer):
    """Enhanced Mixture of Experts with load balancing."""
    
    def __init__(self, d_model, d_ff, num_experts=8, k=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.k = k
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Create experts
        self.experts = []
        for _ in range(num_experts):
            expert = keras.Sequential([
                layers.Dense(d_ff, activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(d_model),
                layers.Dropout(dropout),
            ])
            self.experts.append(expert)
        
        self.gate = layers.Dense(num_experts)
        self.load_balancing_loss_weight = 0.01

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x_flat = tf.reshape(x, [-1, self.d_model])
        num_tokens = tf.shape(x_flat)[0]
        
        # Gate computation
        gate_logits = self.gate(x_flat)
        gate_logits = gate_logits - tf.reduce_max(gate_logits, axis=-1, keepdims=True)
        
        # Top-k gating
        gate_top_k, gate_idx = tf.math.top_k(gate_logits, k=self.k)
        gate_scores = tf.nn.softmax(gate_top_k, axis=-1)
        
        # Load balancing loss
        if training:
            gate_probs = tf.nn.softmax(gate_logits, axis=-1)
            expert_usage = tf.reduce_mean(gate_probs, axis=0)
            cv_squared = tf.reduce_sum(expert_usage ** 2) * self.num_experts
            self.add_loss(self.load_balancing_loss_weight * cv_squared)
        
        # Expert computation
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x_flat, training=training)
            expert_outputs.append(expert_out)
        expert_outputs = tf.stack(expert_outputs, axis=1)
        
        # Gather top-k expert outputs
        batch_indices = tf.range(num_tokens)
        batch_indices = tf.expand_dims(batch_indices, 1)
        batch_indices = tf.tile(batch_indices, [1, self.k])
        gather_idx = tf.stack([batch_indices, gate_idx], axis=-1)
        
        topk_expert_outputs = tf.gather_nd(expert_outputs, gather_idx)
        gate_scores_exp = tf.expand_dims(gate_scores, -1)
        moe_out = tf.reduce_sum(topk_expert_outputs * gate_scores_exp, axis=1)
        
        moe_out = tf.reshape(moe_out, [batch_size, seq_len, self.d_model])
        return moe_out

@tf.recompute_grad
def gradient_checkpoint_block(block_fn, *args, **kwargs):
    """Wrapper for gradient checkpointing."""
    return block_fn(*args, **kwargs)

class TransformerDecoderBlock(layers.Layer):
    """Enhanced Transformer decoder block with gradient checkpointing support."""
    
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        
        if config.use_flash_attention:
            self.attn = FlashAttention(
                config.embed_dim, 
                config.num_heads, 
                config.block_size
            )
        else:
            self.attn = layers.MultiHeadAttention(
                config.num_heads, 
                config.embed_dim // config.num_heads,
                dropout=config.dropout
            )
        
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn = MixtureOfExperts(
            config.embed_dim, 
            config.ffn_dim, 
            config.num_experts, 
            dropout=config.dropout
        )
        
        self.rotary_emb = RotaryEmbedding(config.embed_dim, config.max_seq_len)

    def call(self, x, training=None):
        if self.config.use_gradient_checkpointing and training:
            return gradient_checkpoint_block(self._forward, x, training=training)
        else:
            return self._forward(x, training=training)
    
    def _forward(self, x, training=None):
        # Self-attention with residual connection
        h = self.ln1(x)
        
        if self.config.use_flash_attention:
            attn_out = self.attn(h, training=training)
        else:
            # Apply rotary embeddings for standard attention
            h_rope = self.rotary_emb(h)
            attn_out = self.attn(h_rope, h_rope, training=training)
        
        x = x + attn_out
        
        # Feed-forward with residual connection
        h = self.ln2(x)
        ffn_out = self.ffn(h, training=training)
        return x + ffn_out

class SentencePieceTokenizer:
    """Advanced tokenizer using SentencePiece."""
    
    def __init__(self, vocab_size=32000, model_type='bpe'):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.sp = None
        
    def train(self, texts: List[str], model_path: str = 'tokenizer'):
        """Train SentencePiece tokenizer on texts."""
        # Write texts to temporary file
        with open('temp_corpus.txt', 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # Train SentencePiece model
        sptokenizer.SentencePieceTrainer.train(
            input='temp_corpus.txt',
            model_prefix=model_path,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=['<HUMAN>', '<AI>']
        )
        
        # Load trained model
        self.sp = sptokenizer.SentencePieceProcessor()
        self.sp.load(f'{model_path}.model')
        
        # Clean up
        os.remove('temp_corpus.txt')
        
    def load(self, model_path: str):
        """Load pre-trained tokenizer."""
        self.sp = sptokenizer.SentencePieceProcessor()
        self.sp.load(model_path)
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.sp is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.sp.encode_as_ids(text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.sp is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.sp.decode_ids(ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.sp is None:
            return self.vocab_size
        return self.sp.get_piece_size()

class EnhancedMiniGPT(tf.keras.Model):
    """Enhanced MiniGPT with advanced features."""
    
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Token embedding
        self.embedding = layers.Embedding(
            config.vocab_size, 
            config.embed_dim,
            embeddings_initializer='glorot_uniform'
        )
        
        # Transformer blocks
        self.blocks = [
            TransformerDecoderBlock(config, name=f'block_{i}')
            for i in range(config.num_layers)
        ]
        
        # Final layer norm and output head
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)
        self.head = layers.Dense(config.vocab_size, use_bias=False, dtype='float32')

    def call(self, x, training=None):
        # Ensure input is int32
        x = tf.cast(x, tf.int32)
        
        # Token embeddings
        h = self.embedding(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, training=training)
        
        # Final layer norm and output projection
        h = self.ln_f(h)
        logits = self.head(h)
        
        return tf.cast(logits, tf.float32)
    
    @tf.function
    def generate_step(self, input_ids, temperature=1.0, top_k=40, top_p=0.9):
        """Single generation step."""
        logits = self(input_ids, training=False)
        next_token_logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k > 0:
            top_k_values, _ = tf.math.top_k(next_token_logits, k=top_k)
            min_top_k = top_k_values[:, -1:]
            next_token_logits = tf.where(
                next_token_logits < min_top_k,
                tf.ones_like(next_token_logits) * -1e9,
                next_token_logits
            )
        
        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = tf.nn.top_k(
                next_token_logits, k=tf.shape(next_token_logits)[-1]
            )
            cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits), axis=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove = tf.concat([
                tf.zeros_like(sorted_indices_to_remove[:, :1]),
                sorted_indices_to_remove[:, :-1]
            ], axis=-1)
            
            indices_to_remove = tf.scatter_nd(
                tf.expand_dims(sorted_indices, -1),
                sorted_indices_to_remove,
                tf.shape(next_token_logits)
            )
            next_token_logits = tf.where(
                indices_to_remove,
                tf.ones_like(next_token_logits) * -1e9,
                next_token_logits
            )
        
        # Sample next token
        probs = tf.nn.softmax(next_token_logits)
        next_token = tf.random.categorical(tf.math.log(probs + 1e-8), num_samples=1)
        
        return next_token
    
    def generate_text(self, input_ids, max_length=100, temperature=1.0, top_k=40, top_p=0.9):
        """Generate text with advanced sampling."""
        generated = input_ids
        
        for _ in range(max_length):
            # Trim context if too long
            if tf.shape(generated)[1] >= self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len//2:]
            
            next_token = self.generate_step(generated, temperature, top_k, top_p)
            generated = tf.concat([generated, next_token], axis=1)
            
            # Stop on end token
            if next_token[0][0] == 3:  # EOS token
                break
        
        return generated

class EvaluationSuite:
    """Comprehensive evaluation suite for language models."""
    
    def __init__(self, model: EnhancedMiniGPT, tokenizer: SentencePieceTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def perplexity(self, texts: List[str]) -> float:
        """Calculate perplexity on test texts."""
        total_loss = 0.0
        total_tokens = 0
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) < 2:
                continue
                
            input_ids = tf.constant([tokens[:-1]], dtype=tf.int32)
            targets = tf.constant([tokens[1:]], dtype=tf.int32)
            
            logits = self.model(input_ids, training=False)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets, logits=logits
            )
            
            total_loss += tf.reduce_sum(loss).numpy()
            total_tokens += len(tokens) - 1
        
        avg_loss = total_loss / total_tokens
        return np.exp(avg_loss)
    
    def bleu_score(self, references: List[str], candidates: List[str]) -> float:
        """Calculate BLEU score (simplified implementation)."""
        def n_gram_precision(ref_tokens, cand_tokens, n):
            if len(cand_tokens) < n:
                return 0.0
            
            ref_ngrams = {}
            for i in range(len(ref_tokens) - n + 1):
                ngram = tuple(ref_tokens[i:i+n])
                ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
            
            cand_ngrams = {}
            for i in range(len(cand_tokens) - n + 1):
                ngram = tuple(cand_tokens[i:i+n])
                cand_ngrams[ngram] = cand_ngrams.get(ngram, 0) + 1
            
            matches = 0
            for ngram, count in cand_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            
            return matches / max(len(cand_tokens) - n + 1, 1)
        
        total_score = 0.0
        for ref, cand in zip(references, candidates):
            ref_tokens = ref.split()
            cand_tokens = cand.split()
            
            # Calculate precision for n-grams 1 to 4
            precisions = []
            for n in range(1, 5):
                precisions.append(n_gram_precision(ref_tokens, cand_tokens, n))
            
            # Geometric mean of precisions
            if all(p > 0 for p in precisions):
                bleu = np.exp(np.mean(np.log(precisions)))
                
                # Brevity penalty
                bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))
                bleu *= bp
            else:
                bleu = 0.0
            
            total_score += bleu
        
        return total_score / len(references)
    
    def generation_quality(self, prompts: List[str], num_samples=5) -> Dict:
        """Evaluate generation quality metrics."""
        all_generations = []
        
        for prompt in prompts:
            prompt_tokens = self.tokenizer.encode(prompt)
            input_ids = tf.constant([prompt_tokens], dtype=tf.int32)
            
            for _ in range(num_samples):
                generated = self.model.generate_text(
                    input_ids, max_length=50, temperature=0.8
                )
                generated_text = self.tokenizer.decode(generated[0].numpy().tolist())
                all_generations.append(generated_text)
        
        # Calculate diversity metrics
        unique_generations = set(all_generations)
        diversity = len(unique_generations) / len(all_generations)
        
        # Average length
        avg_length = np.mean([len(text.split()) for text in all_generations])
        
        return {
            "diversity": diversity,
            "average_length": avg_length,
            "total_generations": len(all_generations),
            "unique_generations": len(unique_generations)
        }
    
    def benchmark_speed(self, batch_sizes=[1, 4, 8], seq_lengths=[128, 256, 512]) -> Dict:
        """Benchmark model inference speed."""
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                key = f"batch_{batch_size}_seq_{seq_len}"
                
                # Create dummy input
                dummy_input = tf.random.uniform(
                    (batch_size, seq_len), 
                    maxval=self.tokenizer.get_vocab_size(), 
                    dtype=tf.int32
                )
                
                # Warmup
                for _ in range(3):
                    _ = self.model(dummy_input, training=False)
                
                # Benchmark
                start_time = time.time()
                num_runs = 10
                for _ in range(num_runs):
                    _ = self.model(dummy_input, training=False)
                
                elapsed = time.time() - start_time
                tokens_per_second = (batch_size * seq_len * num_runs) / elapsed
                
                results[key] = {
                    "tokens_per_second": tokens_per_second,
                    "elapsed_time": elapsed / num_runs,
                    "batch_size": batch_size,
                    "sequence_length": seq_len
                }
        
        return results

def create_training_dataset(texts: List[str], tokenizer: SentencePieceTokenizer, 
                          max_length: int = 512, batch_size: int = 8):
    """Create training dataset with proper preprocessing."""
    
    def tokenize_function(text):
        tokens = tokenizer.encode(text.numpy().decode('utf-8'))
        
        # Truncate or pad to max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([0] * (max_length - len(tokens)))  # Pad with 0
            
        return tokens
    
    def py_tokenize(text):
        return tf.py_function(tokenize_function, [text], tf.int32)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    dataset = dataset.map(py_tokenize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_optimizer_with_schedule(config: ModelConfig, steps_per_epoch: int):
    """Create optimizer with learning rate schedule and mixed precision."""
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=steps_per_epoch * 100,  # 100 epochs
        alpha=0.1
    )
    
    # Optimizer with mixed precision
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.95,
        epsilon=1e-8
    )
    
    # Wrap optimizer for mixed precision
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    return optimizer

# Example usage and testing
if __name__ == "__main__":
    # Create model configuration
    config = ModelConfig(
        vocab_size=32000,
        max_seq_len=512,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        ffn_dim=2048,
        num_experts=4,
        use_flash_attention=True,
        use_gradient_checkpointing=True
    )
    
    # Initialize model
    model = EnhancedMiniGPT(config)
    
    # Sample data for testing
    sample_texts = [
        "Hello, how are you today?",
        "The weather is nice outside.",
        "Machine learning is fascinating.",
        "I love programming in Python.",
        "Natural language processing is amazing."
    ]
    
    # Initialize and train tokenizer
    tokenizer = SentencePieceTokenizer(vocab_size=config.vocab_size)
    # Note: In practice, you would use a much larger corpus
    # tokenizer.train(sample_texts, 'my_tokenizer')
    
    # Create evaluation suite
    # evaluator = EvaluationSuite(model, tokenizer)
    
    print("Enhanced MiniGPT model created successfully!")
    print(f"Model parameters: {model.count_params():,}")
    print(f"Configuration: {config}")