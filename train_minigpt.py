import tensorflow as tf
import logging
import numpy as np
import os
from datasets import load_dataset
import random
import json
import math
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from minigpt_transformer import MoEMiniGPT, MoEConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_synthetic_tf_dataset(synthetic_texts, tokenizer, config, repeat=100):
    def synthetic_gen():
        for text in synthetic_texts * repeat:
            tokens = tokenizer.encode(
                text,
                max_length=config.seq_len,
                truncation=True,
                padding='max_length'
            )
            yield {"input_ids": np.array(tokens, dtype=np.int32)}
    return tf.data.Dataset.from_generator(
        synthetic_gen,
        output_signature={"input_ids": tf.TensorSpec(shape=(config.seq_len,), dtype=tf.int32)}
    )

if __name__ == "__main__":
    try:
        # Load custom tokenizer for data encoding
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=ByteLevelBPETokenizer(
                vocab="my-10k-bpe-tokenizer/vocab.json",
                merges="my-10k-bpe-tokenizer/merges.txt",
            ),
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            mask_token="<mask>",
        )

        config = MoEConfig(
            vocab_size=10000,
            max_seq_len=1024,
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            ffn_dim=3072,
            dropout=0.1,
            layer_norm_epsilon=1e-5,
            use_rotary_embeddings=True,
            learning_rate=1e-4,
            batch_size=1,
            seq_len=1024,
            num_experts=8,
            top_k_experts=2,
            use_moe_layers=[2, 4, 6, 8, 10]
        )

        logger.info("Initializing MoEMiniGPT model...")
        model = MoEMiniGPT(config)
        if tokenizer is None:
            raise RuntimeError("MoEMiniGPT tokenizer is not available. Please ensure transformers is installed.")

        # Build the model
        dummy_input = tf.ones((1, config.seq_len), dtype=tf.int32)
        _ = model(dummy_input)  # Build the model

        # Display total number of model parameters
        total_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        logger.info(f"Total model parameters: {total_params:,}")

        # Helper to tokenize and format dataset
        def encode(example):
            if 'text' in example and isinstance(example['text'], str):
                text = example['text']
            elif 'utterance' in example and isinstance(example['utterance'], str):
                text = example['utterance']
            elif 'dialog' in example:
                # If dialog is a list of dicts, extract 'text' from each
                if isinstance(example['dialog'], list):
                    if all(isinstance(turn, dict) and 'text' in turn for turn in example['dialog']):
                        text = " ".join(turn['text'] for turn in example['dialog'])
                    else:
                        text = " ".join(str(turn) for turn in example['dialog'])
                else:
                    text = str(example['dialog'])
            else:
                raise ValueError(f"Cannot find a valid text field in example: {example}")
            if tokenizer is None:
                raise RuntimeError("Tokenizer is not initialized. Please ensure the model provides a valid tokenizer.")
            tokens = tokenizer.encode(
                text,
                max_length=config.seq_len,
                truncation=True,
                padding='max_length'
            )
            return {'input_ids': np.array(tokens, dtype=np.int32)}

        # Load and preprocess ConvAI2
        logger.info("Loading ConvAI2...")
        try:
            convai2 = load_dataset("conv_ai_2", split="train[:1000]")
            sample_convai2 = next(iter(convai2))
            logger.info(f"Sample ConvAI2 example: {sample_convai2}")
            convai2 = convai2.map(encode)
            convai2_tf = tf.data.Dataset.from_generator(
                lambda: ({"input_ids": ex["input_ids"]} for ex in list(convai2)),
                output_signature={"input_ids": tf.TensorSpec(shape=(config.seq_len,), dtype=tf.int32)}
            )
        except Exception as e:
            logger.warning(f"Could not load ConvAI2 dataset: {e}. Using synthetic data instead.")
            synthetic_texts = [
                "Hello, how are you today?",
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming the world.",
                "Let's build a language model together!",
                "What is your favorite programming language?",
                "Deep learning enables powerful models.",
                "This is a synthetic training example.",
                "MoE models can scale efficiently.",
                "Natural language processing is fascinating.",
                "TensorFlow and PyTorch are popular frameworks."
            ]
            convai2_tf = make_synthetic_tf_dataset(synthetic_texts, tokenizer, config)

        # Load and preprocess DailyDialog
        logger.info("Loading DailyDialog...")
        try:
            dailydialog = load_dataset("daily_dialog", split="train[:1000]")
            sample_dailydialog = next(iter(dailydialog))
            logger.info(f"Sample DailyDialog example: {sample_dailydialog}")
            dailydialog = dailydialog.map(lambda ex: {"text": " ".join(ex["dialog"])})
            dailydialog = dailydialog.map(encode)
            dailydialog_tf = tf.data.Dataset.from_generator(
                lambda: ({"input_ids": ex["input_ids"]} for ex in list(dailydialog)),
                output_signature={"input_ids": tf.TensorSpec(shape=(config.seq_len,), dtype=tf.int32)}
            )
        except Exception as e:
            logger.warning(f"Could not load DailyDialog dataset: {e}. Using synthetic data instead.")
            synthetic_texts = [
                "Hi! How can I help you?",
                "Tell me a joke.",
                "What is the weather like today?",
                "Let's discuss machine learning.",
                "Do you like open source software?",
                "This is another synthetic example.",
                "Language models are fun to train.",
                "How do you use attention mechanisms?",
                "Explain transformers in simple terms.",
                "Goodbye and have a nice day!"
            ]
            dailydialog_tf = make_synthetic_tf_dataset(synthetic_texts, tokenizer, config)

        # Combine datasets and batch
        train_dataset = convai2_tf.concatenate(dailydialog_tf).shuffle(2048).batch(config.batch_size)

        # Metrics
        train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_perplexity_metric = tf.keras.metrics.Mean(name='train_perplexity')

        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

        @tf.function
        def train_step(batch):
            input_ids = batch['input_ids']
            targets = input_ids[:, 1:]
            inputs = input_ids[:, :-1]
            with tf.GradientTape() as tape:
                logits, aux_losses = model(inputs, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    targets, logits, from_logits=True
                )
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = 0  # Default to 0 if pad_token_id is not set
                mask = tf.cast(tf.not_equal(targets, pad_token_id), tf.float32)
                mask_sum = tf.reduce_sum(mask)
                loss = tf.reduce_sum(loss * mask) / (mask_sum + 1e-8)  # Prevent division by zero
                if aux_losses:
                    loss += tf.add_n([v for v in aux_losses.values()])
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)  # Gradient clipping
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss_metric.update_state(loss)
            train_accuracy_metric.update_state(targets, logits, sample_weight=mask)
            train_perplexity_metric.update_state(tf.exp(loss))
            return loss

        # Training loop with progress bar and metrics
        logger.info("Starting training...")
        epochs = 5
        steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
        logger.info(f"Epochs: {epochs}, Steps per epoch: {steps_per_epoch}")

        global_step = 0
        for epoch in range(epochs):
            train_loss_metric.reset_state()
            train_accuracy_metric.reset_state()
            train_perplexity_metric.reset_state()
            progbar = tqdm(train_dataset, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
            for batch in progbar:
                global_step += 1
                loss = train_step(batch)
                loss_val = train_loss_metric.result().numpy()
                acc_val = train_accuracy_metric.result().numpy()
                perplexity_val = train_perplexity_metric.result().numpy()
                progbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "acc": f"{acc_val:.4f}",
                    "ppl": f"{perplexity_val:.2f}"
                })
            logger.info(
                f"Epoch {epoch+1}/{epochs} - Loss: {loss_val:.4f} | "
                f"Accuracy: {acc_val:.4f} | Perplexity: {perplexity_val:.2f}"
            )

        # Save model weights (must end with .weights.h5)
        save_dir = "trained_models"
        os.makedirs(save_dir, exist_ok=True)
        weights_path = os.path.join(save_dir, "moe_minigpt.weights.h5")
        model.save_weights(weights_path)
        logger.info(f"Model weights saved to {weights_path}")

        # Save config
        config_path = os.path.join(save_dir, "moe_config.json")
        with open(config_path, 'w') as f:
            config_dict = {k: str(v) if isinstance(v, (list, type(None))) else v for k, v in vars(config).items()}
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")

        # Final generation test (if tokenizer available)
        if model.tokenizer is not None:
            prompt = "Human: What is a mixture of experts model?"
            generated = model.generate_text(prompt, max_length=50)
            logger.info(f"Prompt: {prompt}\nGenerated: {generated}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise