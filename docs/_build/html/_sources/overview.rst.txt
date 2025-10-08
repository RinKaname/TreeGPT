Project Overview
================

TreeGPT is a novel sequence-to-sequence model that replaces the standard attention mechanism with a `TreeFFN` layer. This approach is inspired by the ideas presented in the paper "GCoT: Chain-of-Thought Prompt Learning for Graphs," which proposes a method for iterative reasoning in graph-based models.

The core of TreeGPT is the `TreeFFN`, a graph neural network layer that utilizes message passing and a learnable number of iterations to process sequences. This allows the model to dynamically adjust its computational depth based on the complexity of the task.

This project provides an implementation of TreeGPT and demonstrates its application to the Abstraction and Reasoning Corpus (ARC).