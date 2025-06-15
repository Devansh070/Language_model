import tensorflow as tf
import logging
import numpy as np
import os
from datasets import load_dataset
import random
import json

from minigpt_transformer import MoEMiniGPT, MoEConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def enhanced_train_model(model, config, train_dataset=None, val_dataset=None, epochs=1):
    """Train the MoEMiniGPT model with auxiliary losses."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            input_ids = batch['input_ids']
            labels = batch.get('labels', None)
            loss = model.compute_loss_method(input_ids, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss_metric.update_state(loss)
        return loss

    @tf.function
    def val_step(batch):
        input_ids = batch['input_ids']
        labels = batch.get('labels', None)
        loss = model.compute_loss_method(input_ids, labels)
        val_loss_metric.update_state(loss)
        return loss

    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        train_loss_metric.reset_state()
        val_loss_metric.reset_state()

        # Training loop
        if train_dataset is not None:
            for batch in train_dataset:
                train_step(batch)
        else:
            logger.warning("train_dataset is None. Skipping training loop.")

        # Validation loop
        if val_dataset is not None:
            for batch in val_dataset:
                val_step(batch)
            val_loss = val_loss_metric.result().numpy()
        else:
            val_loss = None

        train_loss = train_loss_metric.result().numpy()
        logger.info(f"Train Loss: {train_loss:.4f}" + (f" | Val Loss: {val_loss:.4f}" if val_loss is not None else ""))
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss if val_loss is not None else float('nan'))

    return model, history

if __name__ == "__main__":
    try:
        config = MoEConfig(
            vocab_size=50257,
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
        tokenizer = model.tokenizer
        if tokenizer is None:
            raise RuntimeError("MoEMiniGPT tokenizer is not available. Please ensure transformers is installed.")

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
        convai2 = load_dataset("conv_ai_2", split="train[:1000]")
        sample_convai2 = next(iter(convai2))
        logger.info(f"Sample ConvAI2 example: {sample_convai2}")
        convai2 = convai2.map(encode)
        convai2_tf = tf.data.Dataset.from_generator(
            lambda: ({"input_ids": ex["input_ids"]} for ex in list(convai2)),
            output_signature={"input_ids": tf.TensorSpec(shape=(config.seq_len,), dtype=tf.int32)}
        )

        # Load and preprocess DailyDialog
        logger.info("Loading DailyDialog...")
        dailydialog = load_dataset("daily_dialog", split="train[:1000]")
        sample_dailydialog = next(iter(dailydialog))
        logger.info(f"Sample DailyDialog example: {sample_dailydialog}")
        dailydialog = dailydialog.map(lambda ex: {"text": " ".join(ex["dialog"])})
        dailydialog = dailydialog.map(encode)
        dailydialog_tf = tf.data.Dataset.from_generator(
            lambda: ({"input_ids": ex["input_ids"]} for ex in list(dailydialog)),
            output_signature={"input_ids": tf.TensorSpec(shape=(config.seq_len,), dtype=tf.int32)}
        )

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
                loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
                if aux_losses:
                    loss += tf.add_n([v for v in aux_losses.values()])
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss_metric.update_state(loss)
            train_accuracy_metric.update_state(targets, logits, sample_weight=mask)
            train_perplexity_metric.update_state(tf.exp(loss))
            return loss

        # Training loop
        logger.info("Starting training...")
        steps = 0
        for epoch in range(1):
            train_loss_metric.reset_state()
            train_accuracy_metric.reset_state()
            train_perplexity_metric.reset_state()
            for batch in train_dataset:
                steps += 1
                loss = train_step(batch)
                if steps % 20 == 0:
                    logger.info(
                        f"Step {steps} | "
                        f"Loss: {train_loss_metric.result():.4f} | "
                        f"Accuracy: {train_accuracy_metric.result():.4f} | "
                        f"Perplexity: {train_perplexity_metric.result():.4f}"
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