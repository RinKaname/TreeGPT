# TreeFFN Documentation

The `TreeFFN.py` script defines the `TreeFFN` class, a graph neural network layer that performs message passing on a tree structure. It is designed to be a flexible and powerful component for models that need to process hierarchical data.

## `TreeFFN` Class

The `TreeFFN` class implements a "Global Parent-Child Aggregation MLP." It processes tree-structured data by aggregating information between parent and child nodes.

### Key Features:

- **Soft Iterations**: It uses a learnable parameter `T` to control the number of message-passing iterations, allowing the model to learn the optimal processing depth.
- **Configurable Architecture**: It supports optional features like edge projection, gating mechanisms, residual connections, and bidirectional processing.
- **Node and Tree-level Outputs**: It can produce outputs for both individual nodes and the entire tree.

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `d_in` | `int` | The feature dimensionality of the input nodes. |
| `d_h` | `int` | The dimensionality of the hidden layers. |
| `num_node_classes` | `int` | The number of output classes for node-level predictions. If `None`, no node-level output is produced. |
| `num_tree_classes` | `int` | The number of output classes for tree-level predictions. If `None`, no tree-level output is produced. |
| `use_edge_proj` | `bool` | If `True`, applies a linear projection to the concatenated parent-child features. |
| `use_gating` | `bool` | If `True`, uses a gating mechanism to control the message flow between nodes. |
| `residual` | `bool` | If `True`, adds a residual connection from the input to the output of the layer. |
| `bidirectional` | `bool` | If `True`, processes the tree in both top-down and bottom-up directions. |
| `dropout` | `float` | The dropout rate applied to the final hidden state. |
| `tree_iterations` | `int` | The initial value for the learnable parameter `T` that controls the number of soft iterations. |

### Forward Pass

The `forward` method executes the message-passing algorithm. Here is a step-by-step breakdown:

1. **Initialization**: The input node features are projected into a hidden space `h`.
2. **Soft Iterations**: The model performs a fixed number of `max_iterations`, but the contribution of each iteration is weighted by `torch.sigmoid(self.T - step_idx)`. This makes the number of iterations "soft" and learnable.
3. **Message Calculation**: For each edge, a message is computed. This can be a simple sum of parent and child hidden states or a more complex projection (`use_edge_proj`). A gating mechanism (`use_gating`) can also be applied.
4. **Aggregation**: The messages are aggregated at each node. Each node sums the messages from all its connections (both as a parent and as a child).
5. **State Update**: The aggregated messages are used to update the node's hidden state. This step includes a ReLU activation and an optional residual connection.
6. **Accumulation**: The hidden state from each iteration is weighted and accumulated.
7. **Output**: The final hidden state is produced after applying dropout to the accumulated hidden states. If `num_node_classes` or `num_tree_classes` are set, the model computes and returns the corresponding logits.