# ARC-TreeGPT Documentation

The `arc_treegpt.py` script provides the necessary components to adapt the `TreeGPT` model for the Abstraction and Reasoning Corpus (ARC) dataset. This includes a specialized tokenizer, a PyTorch Dataset class, and a wrapper for the `TreeGPT` model.

## `ARCGridTokenizer` Class

This class is responsible for converting the 2D grids of the ARC dataset into 1D sequences of tokens, and vice-versa.

### Key Features

-   **Vocabulary**: It defines a small vocabulary that includes the 10 ARC colors (0-9) and several special tokens for structuring the data.
-   **Special Tokens**:
    -   `GRID_START` / `GRID_END`: Mark the beginning and end of a grid.
    -   `ROW_SEP`: Separates rows within a grid.
    -   `EXAMPLE_SEP`: Separates different training examples in the prompt.
    -   `INPUT_OUTPUT_SEP`: Separates the input and output grids of a training example.
    -   `TEST_START`: Marks the beginning of the test input grid.
    -   `PAD`: A padding token.
-   **Encoding**: The `encode_arc_sample` method constructs a single sequence from an ARC problem's training examples and test input. The format is: `exam_in<sep>exam_out<sep>...<sep>test_in`.
-   **Decoding**: The `decode_grid` method converts a sequence of tokens back into a 2D grid.

## `ARCDataset` Class

This PyTorch `Dataset` class loads the ARC data from a JSON file and prepares it for training.

### Key Features

-   **Dynamic Length**: It handles sequences of varying lengths without enforcing a fixed size for all samples.
-   **Input and Target Creation**: For each sample, it creates a single sequence containing both the input prompt and the target output grid.
-   **No Padding**: The dataset itself does not pad the sequences. Padding is handled by the `collate_fn` at the batch level.

## `ARCTreeGPT` Class

This class is a simple wrapper around the base `TreeGPT` model, tailored for the ARC task.

### `generate_arc_solution` Method

This method is used for inference. It takes an input sequence and generates a solution grid.

-   **Parallel Inference**: It performs a single forward pass of the `TreeGPT` model to get logits for the entire output sequence at once.
-   **Decoding**: It takes the `argmax` of the logits to get the predicted token sequence and then uses the `ARCGridTokenizer` to decode the output part of the sequence back into a 2D grid.

## `collate_fn` Function

This custom collate function is used with a `DataLoader`. Its main responsibility is to pad the sequences within a batch to the same length.

-   **Dynamic Padding**: It finds the maximum sequence length within a batch and pads all shorter sequences to that length using the `PAD` token. This is more efficient than padding all sequences in the dataset to a global maximum length.