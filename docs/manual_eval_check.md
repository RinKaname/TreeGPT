# manual_eval_check.py Documentation

The `manual_eval_check.py` script is a utility for manually evaluating the performance of a trained `TreeGPT` model on the ARC dataset. It loads the best-performing model checkpoint and runs it on a few randomly selected samples from the evaluation set.

## Key Functionality

-   **Load Best Model**: The `load_best_model` function loads the model checkpoint saved at `best_treeffn_seq2seq.pth`. It restores the model's architecture, weights, and training configuration.
-   **Manual Evaluation**: The `manual_evaluation` function performs the following steps:
    1.  Loads the ARC evaluation dataset.
    2.  Selects a few random samples from the dataset.
    3.  For each sample, it runs the model to get predictions.
    4.  It calculates and prints the token-level and full-sequence accuracy.
    5.  It displays a comparison of the predicted tokens and the target tokens for a detailed inspection.
    6.  It attempts to decode the task structure to provide context, such as the size of the input and output grids.

## How to Use

To run the script, simply execute it from the command line:

```bash
python manual_eval_check.py
```

The script will print detailed logs of the evaluation process for a small number of samples, allowing for a qualitative assessment of the model's performance.