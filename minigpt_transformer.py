import tensorflow as tf
import numpy as np
keras = tf.keras
layers = tf.keras.layers

class RotaryEmbedding(layers.Layer):
    def __init__(self, dim=None, max_seq_len=256, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len

    def build(self, input_shape):
        if self.dim is None:
            self.dim = int(input_shape[-1])
        assert self.dim % 2 == 0, "RotaryEmbedding: dim must be even"
        inv_freq = 1.0 / (10000 ** (np.arange(0, self.dim // 2) / (self.dim // 2)))
        t = np.arange(self.max_seq_len)
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

def causal_attention_mask(batch_size, seq_len):
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    mask = tf.reshape(mask, (1, 1, seq_len, seq_len))
    return tf.tile(mask, [batch_size, 1, 1, 1])

class SparseSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, block_size=128, rotary_emb=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = embed_dim // num_heads
        self.q_proj = layers.Dense(embed_dim, use_bias=False)
        self.k_proj = layers.Dense(embed_dim, use_bias=False)
        self.v_proj = layers.Dense(embed_dim, use_bias=False)
        self.out_proj = layers.Dense(embed_dim, use_bias=False)
        self.rotary_emb = rotary_emb

    def call(self, x, attention_mask=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        def split_heads(t):
            t = tf.reshape(t, [batch_size, seq_len, self.num_heads, self.head_dim])
            return tf.transpose(t, [0, 2, 1, 3])
        
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        
        if self.rotary_emb is not None:
            def apply_rope(t):
                shape = tf.shape(t)
                t_ = tf.reshape(t, [-1, seq_len, self.head_dim])
                t_ = self.rotary_emb(t_)
                return tf.reshape(t_, shape)
            q = apply_rope(q)
            k = apply_rope(k)
        
        attn_scores = tf.matmul(q, k, transpose_b=True)
        attn_scores = attn_scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        window = self.block_size
        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]
        local_mask = tf.cast((j <= i) & (j >= i - window + 1), tf.float32)
        local_mask = tf.reshape(local_mask, (1, 1, seq_len, seq_len))
        
        attn_scores = attn_scores * local_mask
        attn_scores = tf.where(
            tf.equal(local_mask, 0.0),
            tf.ones_like(attn_scores) * (-1e9),
            attn_scores
        )
        
        attn_scores = attn_scores - tf.reduce_max(attn_scores, axis=-1, keepdims=True)
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        
        attn_output = tf.matmul(attn_weights, v)
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [batch_size, seq_len, self.embed_dim])
        
        out = self.out_proj(attn_output)
        return out + x

class MoE(layers.Layer):
    def __init__(self, d_model, d_ff, num_experts=4, k=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.k = k
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.experts = [
            keras.Sequential([
                layers.Dense(d_ff, activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(d_model),
                layers.Dropout(dropout),
            ]) for _ in range(num_experts)
        ]
        self.gate = layers.Dense(num_experts)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x_flat = tf.reshape(x, [-1, self.d_model])
        num_tokens = tf.shape(x_flat)[0]
        
        gate_logits = self.gate(x_flat)
        gate_logits = gate_logits - tf.reduce_max(gate_logits, axis=-1, keepdims=True)
        
        gate_top_k, gate_idx = tf.math.top_k(gate_logits, k=self.k)
        gate_scores = tf.nn.softmax(gate_top_k, axis=-1)
        
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x_flat, training=training)
            expert_outputs.append(expert_out)
        expert_outputs = tf.stack(expert_outputs, axis=1)
        
        batch_indices = tf.range(num_tokens)
        batch_indices = tf.expand_dims(batch_indices, 1)
        batch_indices = tf.tile(batch_indices, [1, self.k])
        gather_idx = tf.stack([batch_indices, gate_idx], axis=-1)
        
        topk_expert_outputs = tf.gather_nd(expert_outputs, gather_idx)
        gate_scores_exp = tf.expand_dims(gate_scores, -1)
        moe_out = tf.reduce_sum(topk_expert_outputs * gate_scores_exp, axis=1)
        
        moe_out = tf.reshape(moe_out, [batch_size, seq_len, self.d_model])
        return moe_out + x

def FeedForward(dim, hidden_dim, dropout=0.1):
    return keras.Sequential([
        layers.Dense(hidden_dim, activation='gelu'),
        layers.Dropout(dropout),
        layers.Dense(dim),
        layers.Dropout(dropout),
    ])

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_dim, rotary_emb, dropout=0.1, num_experts=4, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = SparseSelfAttention(embed_dim, num_heads, rotary_emb=rotary_emb)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn = MoE(embed_dim, ffn_dim, num_experts=num_experts, k=2, dropout=dropout)

    def call(self, x, mask=None, training=None):
        h = self.ln1(x)
        attn_out = self.attn(h, attention_mask=mask)
        x = x + attn_out
        h = self.ln2(x)
        ffn_out = self.ffn(h, training=training)
        return x + ffn_out

def load_pretrained_embeddings(vocab_size, embed_dim, embedding_matrix=None):
    if embedding_matrix is not None:
        emb_layer = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim,
            weights=[embedding_matrix], trainable=False
        )
    else:
        emb_layer = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
    return emb_layer

class MiniGPT(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len=512, embed_dim=256, num_heads=8, num_layers=6, ffn_dim=1024, embedding_matrix=None, num_experts=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.embedding = load_pretrained_embeddings(vocab_size, embed_dim, embedding_matrix)
        self.rotary_emb = RotaryEmbedding(embed_dim, max_seq_len)
        self.blocks = [
            TransformerDecoderBlock(embed_dim, num_heads, ffn_dim, self.rotary_emb, num_experts=num_experts)
            for _ in range(num_layers)
        ]
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)
        self.head = layers.Dense(vocab_size, use_bias=False)

    def build(self, input_shape):
        self.embedding.build(input_shape)
        self.rotary_emb.build((None, input_shape[1], self.embed_dim))
        for block in self.blocks:
            block.build((None, input_shape[1], self.embed_dim))
        self.ln_f.build((None, input_shape[1], self.embed_dim))
        self.head.build((None, input_shape[1], self.embed_dim))
        super().build(input_shape)

    def call(self, x, training=None):
        x = tf.cast(x, tf.int32)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        mask = causal_attention_mask(batch_size, seq_len)
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h, mask=mask, training=training)
        h = self.ln_f(h)
        logits = self.head(h)
        return tf.cast(logits, tf.float32)
    
    def generate_text(self, input_ids, max_length=100, temperature=1.0, top_k=40, top_p=0.85):
        generated = input_ids
        for _ in range(max_length):
            logits = self(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = next_token_logits < tf.math.top_k(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = tf.where(indices_to_remove, tf.ones_like(next_token_logits) * -1e9, next_token_logits)
            
            if top_p < 1.0:
                sorted_logits = tf.sort(next_token_logits, direction='DESCENDING')
                sorted_indices = tf.argsort(next_token_logits, direction='DESCENDING')
                cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = tf.concat([tf.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, :-1]], axis=-1)
                indices_to_remove = tf.gather(sorted_indices_to_remove, tf.argsort(sorted_indices), axis=-1)
                next_token_logits = tf.where(indices_to_remove, tf.ones_like(next_token_logits) * -1e9, next_token_logits)
            
            probs = tf.nn.softmax(next_token_logits, axis=-1)
            next_token = tf.random.categorical(probs, num_samples=1)
            generated = tf.concat([generated, next_token], axis=1)
            
            if next_token[0][0] == 0:
                break
        
        return generated

class ChatTokenizer:
    def __init__(self, vocab_size=10000):
        self._max_vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
            '<HUMAN>': 4,
            '<AI>': 5
        }
        for token, idx in self.special_tokens.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

    def fit_on_texts(self, texts):
        if not texts:
            raise ValueError("Empty text list provided")
        word_counts = {}
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            words = text.split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        for word, _ in sorted_words:
            if len(self.word_to_id) >= self._max_vocab_size:
                break
            if word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
        print(f"Vocabulary size: {len(self.word_to_id)}")
        return self

    def texts_to_sequences(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        sequences = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            words = text.split()
            sequence = [self.word_to_id.get(word, self.special_tokens['<UNK>']) for word in words]
            sequences.append(sequence)
        return sequences[0] if len(sequences) == 1 else sequences

    def sequences_to_texts(self, sequences):
        if not isinstance(sequences, list):
            sequences = [sequences]
        texts = []
        for sequence in sequences:
            if not isinstance(sequence, (list, np.ndarray, tf.Tensor)):
                sequence = [sequence]
            if isinstance(sequence, tf.Tensor):
                sequence = sequence.numpy().tolist()
            if isinstance(sequence, np.ndarray):
                sequence = sequence.tolist()
            text = ' '.join(self.id_to_word.get(int(id_), '<UNK>') for id_ in sequence)
            texts.append(text)
        return texts[0] if len(texts) == 1 else texts

    @property
    def vocab_size(self):
        return len(self.word_to_id)