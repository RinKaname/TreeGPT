# train_no_attention.py Documentation

The `train_no_attention.py` script is the main entry point for training a `TreeGPT` model on the ARC dataset. This script is designed to run an experiment with a "pure" `TreeFFN` sequence-to-sequence architecture, completely removing any attention mechanisms or auto-regressive decoding.

## Model Architecture

This script defines its own simplified versions of the `TreeFFNSeq2SeqBlock` and `TreeFFNSeq2Seq` models.

-   **`TreeFFNSeq2SeqBlock`**: This is a simplified block that contains only an encoder and a decoder `TreeFFN`. It does not include the additional feed-forward layer found in the main `TreeGPT` model.
-   **`TreeFFNSeq2Seq`**: This model stacks the simplified `TreeFFNSeq2SeqBlock` layers. It is designed for purely parallel processing of sequences.
-   **`TreeFFNSeq2SeqARC`**: A simple wrapper around `TreeFFNSeq2Seq` to set the correct vocabulary size for the ARC task.

## Training Process

The `train_treeffn_seq2seq_model` function orchestrates the training loop.

### Key Steps:

1.  **Configuration**: It sets up the model configuration, including hyperparameters like `d_model`, `n_layers`, and `tree_iterations`.
2.  **Model and Optimizer**: It initializes the `TreeFFNSeq2SeqARC` model and an AdamW optimizer. A key detail is that it sets a higher learning rate for the learnable `T` parameter in the `TreeFFN` layers, encouraging the model to learn the optimal number of iterations.
3.  **Data Loading**: It loads the ARC training and evaluation datasets using the `ARCDataset` class and a `DataLoader`.
4.  **Training Loop**: The script iterates through the training data for a fixed number of epochs. In each step, it performs the following:
    -   A purely parallel forward pass of the model.
    -   Calculates the cross-entropy loss.
    -   Computes both token-level accuracy and full-sequence accuracy.
    -   Performs backpropagation and updates the model's weights.
5.  **Validation**: Periodically, the model is evaluated on a subset of the evaluation dataset to monitor its performance. The `evaluate_treeffn_seq2seq_model` function is used for this.
6.  **Checkpointing**: The best-performing model (based on validation accuracy) is saved to `best_treeffn_seq2seq.pth`.
7.  **History Logging**: The training progress, including loss and accuracy metrics, is saved to `training_history_treeffn_seq2seq.json`.

## How to Run

To start the training process, simply run the script from the command line:

```bash
python train_no_attention.py
```

The script will log the training progress to the console and save the best model and training history to files.