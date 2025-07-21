import logging
import numpy as np
import os
import json
import math
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import mixed_precision
#Devansh Sinha
from minigpt_transformer import MoEMiniGPT, MoEConfig
#Devansh Sinha
# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mixed_precision.set_global_policy('mixed_float16')
#Devansh Sinha
if __name__ == "__main__":
    try:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file="my-10k-bpe-tokenizer/tokenizer.json",
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            mask_token="<mask>",
        )
#Devansh Sinha
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
            learning_rate=2e-4,
            batch_size=32,
            num_experts=4,
            top_k_experts=1,
            use_moe_layers=[2, 4, 6]
        )
#Devansh Sinha
        logger.info("Initializing MoEMiniGPT model...")
        model = MoEMiniGPT(config)

        dummy_input = tf.ones((1, config.seq_len), dtype=tf.int32)
        _ = model(dummy_input)

        total_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        logger.info(f"Total model parameters: {total_params:,}")

        # Load corpus and tokenize
        corpus_path = "corpus.txt"
        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
#Devansh Sinha
        def encode_line(line):
            tokens = tokenizer.encode(
                line,
                max_length=config.seq_len,
                truncation=True,
                padding='max_length'
            )
            return {"input_ids": np.array(tokens, dtype=np.int32)}

        encoded = [encode_line(line) for line in lines]
#Devansh Sinha
        train_dataset = tf.data.Dataset.from_generator(
            lambda: (ex for ex in encoded),
            output_signature={"input_ids": tf.TensorSpec(shape=(config.seq_len,), dtype=tf.int32)}
        ).shuffle(2048).batch(config.batch_size)

        logger.info(f"Training dataset created with {len(encoded)} examples.")
        total_tokens = sum(len(tokenizer.encode(line)) for line in lines)
        logger.info(f"Total number of tokens in corpus: {total_tokens}")

        train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#Devansh Sinha
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
#Devansh Sinha
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
                pad_token_id = getattr(tokenizer, "pad_token_id", 0)
                mask = tf.cast(tf.not_equal(targets, pad_token_id), tf.float32)
                mask_sum = tf.reduce_sum(mask)
                loss = tf.reduce_sum(loss * mask) / (mask_sum + 1e-8)
                if aux_losses:
                    loss += tf.add_n([v for v in aux_losses.values()])
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss_metric.update_state(loss)
            train_accuracy_metric.update_state(targets, logits, sample_weight=mask)
            return loss
#Devansh Sinha
        logger.info("Starting training...")
        epochs = 2
        steps_per_epoch = math.ceil(len(encoded) / config.batch_size)
        logger.info(f"Epochs: {epochs}, Steps per epoch: {steps_per_epoch}")
#Devansh Sinha
        global_step = 0
        for epoch in range(epochs):
            train_loss_metric.reset_state()
            train_accuracy_metric.reset_state()
            epoch_losses = []
#Devansh Sinha
            logger.info(f"Epoch {epoch+1}/{epochs} started.")
            progbar = tqdm(train_dataset, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
            for step, batch in enumerate(progbar, 1):
                global_step += 1
                loss = train_step(batch)
                epoch_losses.append(loss.numpy())
#Devansh Sinha
                loss_val = train_loss_metric.result().numpy()
                acc_val = train_accuracy_metric.result().numpy()
#Devansh Sinha
                progbar.set_postfix({
                    "step": f"{step}/{steps_per_epoch}",
                    "loss": f"{loss_val:.4f}",
                    "acc": f"{acc_val:.4f}"
                })
#Devansh Sinha
            avg_loss = np.mean(epoch_losses)
            perplexity = math.exp(avg_loss)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | Accuracy: {acc_val:.4f} | Perplexity: {perplexity:.2f}")
#Devansh Sinha
        # Save model
        save_dir = "trained_models"
        os.makedirs(save_dir, exist_ok=True)
        weights_path = os.path.join(save_dir, "moe_minigpt.weights.h5")
        model.save_weights(weights_path)
        logger.info(f"Model weights saved to {weights_path}")
#Devansh Sinha
        config_path = os.path.join(save_dir, "moe_config.json")
        with open(config_path, 'w') as f:
            config_dict = {k: str(v) if isinstance(v, (list, type(None))) else v for k, v in vars(config).items()}
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
#Devansh Sinha
        # Optional chat interface
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
