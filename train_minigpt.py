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
            # Assume batch is a dict with 'input_ids' and optionally 'labels'
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
        train_loss_metric.reset_states()
        val_loss_metric.reset_states()

        # Training loop
        for batch in train_dataset:
            train_step(batch)

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
        dummy_input = tf.random.uniform((1, config.seq_len), maxval=config.vocab_size, dtype=tf.int32)
        _ = model(dummy_input)

        # Dummy dataset for demonstration (replace with real data loader)
        def gen():
            for _ in range(10):
                arr = np.random.randint(0, config.vocab_size, (config.batch_size, config.seq_len), dtype=np.int32)
                yield {'input_ids': arr}
        train_dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature={'input_ids': tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32)}
        ).batch(config.batch_size)

        # Train the model
        trained_model, history = enhanced_train_model(model, config, train_dataset, epochs=1)

        # Save model weights (must end with .weights.h5)
        save_dir = "trained_models"
        os.makedirs(save_dir, exist_ok=True)
        weights_path = os.path.join(save_dir, "moe_minigpt.weights.h5")
        trained_model.save_weights(weights_path)
        logger.info(f"Model weights saved to {weights_path}")

        # Save config
        config_path = os.path.join(save_dir, "moe_config.json")
        with open(config_path, 'w') as f:
            config_dict = {k: str(v) if isinstance(v, (list, type(None))) else v for k, v in vars(config).items()}
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")

        # Final generation test (if tokenizer available)
        if trained_model.tokenizer is not None:
            prompt = "Human: What is a mixture of experts model?"
            generated = trained_model.generate_text(prompt, max_length=50)
            logger.info(f"Prompt: {prompt}\nGenerated: {generated}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise