# TreeGPT Documentation

The `TreeGPT.py` script defines the `TreeGPT` model, a sequence-to-sequence architecture that replaces the standard attention mechanism with a `TreeFFN`-based block.

## `TreeFFNSeq2SeqBlock` Class

This class is the core building block of the `TreeGPT` model. It functions as an alternative to the Transformer's self-attention block, processing sequences in a bidirectional manner.

### Architecture

The `TreeFFNSeq2SeqBlock` consists of two `TreeFFN` instances:

1.  **Encoder `TreeFFN`**: Processes the input sequence from left to right.
2.  **Decoder `TreeFFN`**: Processes the input sequence from right to left.

This bidirectional processing allows the model to capture dependencies from both past and future tokens in the sequence.

### Forward Pass

1.  **Encoder Pass**: The input sequence is first passed through the left-to-right `TreeFFN`.
2.  **Residual Connection**: The output of the encoder is added to the original input.
3.  **Decoder Pass**: The result is then passed through the right-to-left `TreeFFN`.
4.  **Final Residual Connection**: The output of the decoder is added to its input, producing the final output of the block.

Layer normalization is applied before both the encoder and decoder passes.

## `TreeGPT` Class

The `TreeGPT` model is a sequence-to-sequence model designed for tasks where hierarchical or sequential dependencies are important. It is built by stacking multiple `TreeFFNSeq2SeqBlock` layers.

### Key Features

-   **Attention-Free**: It does not use any self-attention mechanism, which can lead to computational savings, especially for long sequences.
-   **Parallel Processing**: The model processes the entire sequence in parallel, unlike auto-regressive models.
-   **Fixed-Depth Processing**: The `TreeFFN` layers use a "soft" number of iterations, allowing the model to learn the optimal processing depth for the task.

### Parameters

| Parameter         | Type    | Description                                           |
| ----------------- | ------- | ----------------------------------------------------- |
| `vocab_size`      | `int`   | The size of the vocabulary.                           |
| `d_model`         | `int`   | The dimensionality of the model's hidden states.      |
| `n_layers`        | `int`   | The number of `TreeFFNSeq2SeqBlock` layers to stack.  |
| `max_seq_len`     | `int`   | The maximum sequence length the model can handle.     |
| `dropout`         | `float` | The dropout rate used in the `TreeFFN` blocks.        |
| `tree_iterations` | `int`   | The initial number of iterations for the `TreeFFN` layers. |

### Forward Pass

1.  **Embeddings**: The input token IDs are converted into token embeddings. Positional embeddings are also added to provide the model with information about the token order.
2.  **`TreeFFNSeq2SeqBlock` Stack**: The combined embeddings are passed through a series of `TreeFFNSeq2SeqBlock` layers.
3.  **Final Layer Norm**: A final layer normalization is applied to the output of the last block.
4.  **Output Head**: A linear layer projects the final hidden state into the vocabulary space, producing the output logits.