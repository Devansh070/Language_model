import logging
import numpy as np
import os
from datasets import load_dataset
import random
import json
import math
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from minigpt_transformer import MoEMiniGPT, MoEConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable mixed precision for better GPU utilization
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

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
            tokenizer_file="my-10k-bpe-tokenizer/tokenizer.json",
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            mask_token="<mask>",
        )
 
        # Increased batch size for better GPU utilization
        config = MoEConfig(
            vocab_size=10000,
            max_seq_len=256,
            seq_len=256,
            embed_dim=512,
            num_heads=8,
            num_layers=8,
            ffn_dim=2048,
            dropout=0.1,
            layer_norm_epsilon=1e-5,
            use_rotary_embeddings=True,
            learning_rate=1e-4,
            batch_size=8,  # Increased from 4 to 8 for better GPU utilization
            num_experts=4,
            top_k_experts=1,
            use_moe_layers=[2,4,6]
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

        # Load and encode the corpus.txt lines
        corpus_path = "corpus.txt"
        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        def encode_line(line):
            tokens = tokenizer.encode(
                line,
                max_length=config.seq_len,
                truncation=True,
                padding='max_length'
            )
            return {"input_ids": np.array(tokens, dtype=np.int32)}

        encoded = [encode_line(line) for line in lines]

        # Optimized dataset pipeline
        train_dataset = tf.data.Dataset.from_generator(
            lambda: (ex for ex in encoded),
            output_signature={"input_ids": tf.TensorSpec(shape=(config.seq_len,), dtype=tf.int32)}
        ).shuffle(buffer_size=10000).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

        logger.info(f"Training dataset created with {len(encoded)} examples.")

        # Count total number of tokens in the corpus
        total_tokens = sum(len(tokenizer.encode(line)) for line in lines)
        logger.info(f"Total number of tokens in corpus: {total_tokens}")

        # Fixed metrics initialization
        train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        # Learning rate schedule for better convergence
        initial_learning_rate = config.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate,
            decay_steps=1000,
            alpha=0.1
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        @tf.function
        def train_step(batch):
            input_ids = batch['input_ids']
            targets = input_ids[:, 1:]
            inputs = input_ids[:, :-1]
            
            with tf.GradientTape() as tape:
                logits, aux_losses = model(inputs, training=True)
                
                # Compute loss
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    targets, logits, from_logits=True
                )
                
                # Apply padding mask
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = 0
                mask = tf.cast(tf.not_equal(targets, pad_token_id), tf.float32)
                mask_sum = tf.reduce_sum(mask)
                
                # Masked loss calculation
                masked_loss = tf.reduce_sum(loss * mask) / (mask_sum + 1e-8)
                
                # Add auxiliary losses
                if aux_losses:
                    aux_loss = tf.add_n([v for v in aux_losses.values()])
                    total_loss = masked_loss + aux_loss
                else:
                    total_loss = masked_loss
                
                # Scale loss for mixed precision
                if policy.compute_dtype == tf.float16:
                    scaled_loss = optimizer.get_scaled_loss(total_loss)
                else:
                    scaled_loss = total_loss
            
            # Compute and apply gradients
            if policy.compute_dtype == tf.float16:
                scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
                grads = optimizer.get_unscaled_gradients(scaled_grads)
            else:
                grads = tape.gradient(scaled_loss, model.trainable_variables)
            
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # Update metrics
            train_loss_metric.update_state(masked_loss)
            train_accuracy_metric.update_state(targets, logits, sample_weight=mask)
            
            return masked_loss

        # Training loop with fixed perplexity calculation
        logger.info("Starting training...")
        epochs = 5
        steps_per_epoch = math.ceil(len(encoded) / config.batch_size)
        logger.info(f"Epochs: {epochs}, Steps per epoch: {steps_per_epoch}")

        global_step = 0
        for epoch in range(epochs):
            train_loss_metric.reset_state()
            train_accuracy_metric.reset_state()
            
            logger.info(f"Epoch {epoch+1}/{epochs} started.")
            progbar = tqdm(train_dataset, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}", ncols=120)
            
            for step, batch in enumerate(progbar, 1):
                global_step += 1
                loss = train_step(batch)
                
                # Get current metrics
                current_loss = train_loss_metric.result().numpy()
                current_acc = train_accuracy_metric.result().numpy()
                current_ppl = np.exp(current_loss)  # Fixed perplexity calculation
                
                # Update progress bar with proper formatting
                progbar.set_postfix({
                    "loss": f"{current_loss:.3f}",
                    "acc": f"{current_acc:.3f}",
                    "ppl": f"{current_ppl:.1f}",
                    "lr": f"{optimizer.learning_rate.numpy():.2e}"
                })
                
                # Log every 1000 steps
                if step % 1000 == 0:
                    logger.info(f"Step {step}: Loss={current_loss:.4f}, Acc={current_acc:.4f}, PPL={current_ppl:.2f}")
            
            # End of epoch logging
            final_loss = train_loss_metric.result().numpy()
            final_acc = train_accuracy_metric.result().numpy()
            final_ppl = np.exp(final_loss)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} completed - "
                f"Loss: {final_loss:.4f} | "
                f"Accuracy: {final_acc:.4f} | "
                f"Perplexity: {final_ppl:.2f}"
            )

        # Save model weights
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

        # Final generation test
        if hasattr(model, 'generate_text'):
            prompt = "Human: What is a mixture of experts model?"
            generated = model.generate_text(prompt, max_length=50)
            logger.info(f"Prompt: {prompt}\nGenerated: {generated}")

        # Interactive chat loop
        if hasattr(model, "generate_text"):
            print("\n--- Chat with your model! Type 'quit' to exit. ---")
            while True:
                user_input = input("You: ")
                if user_input.strip().lower() in ["quit", "exit"]:
                    print("Exiting chat.")
                    break
                response = model.generate_text(user_input, max_length=50)
                print("Model:", response)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise