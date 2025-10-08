Conceptual Guide
================

This guide delves into the core concepts behind TreeGPT, explaining the architecture and the theoretical foundations of the model.

The `TreeFFN` Layer
-------------------

The `TreeFFN` (Tree Feed-Forward Network) is a graph neural network layer that forms the building block of TreeGPT. It operates on a sequence of inputs by treating them as nodes in a graph and performing message passing between them.

Key features of the `TreeFFN` layer include:

*   **Learnable Iterations (`T`)**: The `TreeFFN` layer has a learnable parameter `T` that controls the number of message-passing iterations. This allows the model to learn the optimal computational depth for a given task, which is a core concept inspired by the GCoT paper.
*   **Bidirectional Processing**: The `TreeFFNSeq2SeqBlock` in `TreeGPT` uses two `TreeFFN` instances to process the input sequence in both left-to-right and right-to-left directions, capturing dependencies from both sides.
*   **Gating and Residual Connections**: The layer can incorporate gating mechanisms and residual connections to improve information flow and training stability.

Connection to GCoT
------------------

The GCoT paper, "GCoT: Chain-of-Thought Prompt Learning for Graphs," introduces a framework for enabling graph models to perform step-by-step reasoning. While TreeGPT does not implement the full GCoT framework, the `TreeFFN`'s learnable iteration parameter `T` is a direct application of the paper's core idea. By learning `T`, the model can dynamically decide how many "reasoning steps" (message-passing iterations) are needed to solve a task.

The `TreeGPT` Model
-------------------

The `TreeGPT` model is a sequence-to-sequence architecture that stacks multiple `TreeFFNSeq2SeqBlock` layers. It is a non-autoregressive model, meaning it processes the entire input sequence in parallel, making it potentially faster than traditional autoregressive models like the Transformer.