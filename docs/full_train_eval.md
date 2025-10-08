# full_train_eval.py Documentation

The `experiments/full_train_eval.py` script is a comprehensive pipeline for training and evaluating the `ARCTreeGPT` model on the ARC dataset. It uses a pure `TreeFFN`-based sequence-to-sequence architecture, with no attention or auto-regressive decoding.

This script represents the official implementation for a full training and evaluation cycle.

## Core Pipeline

The script is organized into a clear pipeline, managed by the `pure_treeffn_pipeline` function.

1.  **Train**: It first checks if a trained model (`best_pure_treeffn.pth`) already exists. If not, it calls `train_pure_treeffn_model` to train a new model from scratch.
2.  **Evaluate**: After training (or if a model already existed), it calls `evaluate_model` to run a final evaluation on the test set.

## Key Functions

-   **`train_pure_treeffn_model()`**:
    -   **Model**: Initializes the `ARCTreeGPT` model from the `src` directory.
    -   **Optimizer**: Uses AdamW with a higher learning rate for the `TreeFFN`'s learnable `T` parameter.
    -   **Training Loop**: Trains the model on the ARC training data. It tracks loss, token accuracy, and full-sequence accuracy.
    -   **Validation & Checkpointing**: Periodically evaluates the model on a validation set and saves the best-performing checkpoint to `best_pure_treeffn.pth`.
    -   **Logging**: Saves the complete training history to `training_history_pure_treeffn.json`.

-   **`evaluate_model(model_path)`**:
    -   **Model Loading**: Loads a specified model checkpoint.
    -   **Final Evaluation**: Runs the model on the entire evaluation dataset (`arc-agi_evaluation_challenges.json`).
    -   **Results**: Calculates the final token and full-sequence accuracy and saves these metrics to `evaluation_results.json`.

-   **`pure_treeffn_pipeline()`**:
    -   The main entry point that orchestrates the train-then-evaluate workflow.

## How to Use

To run the complete training and evaluation pipeline, execute the script from the command line:

```bash
python experiments/full_train_eval.py
```

The script will first train the model (if necessary), then evaluate it, and save all results (checkpoints, training history, and final evaluation metrics) to the root directory.