# Week 7: Graph Neural Networks for Recommendations

## Overview

Graph Neural Networks (GNNs) leverage the graph structure of user-item interactions and item relationships to learn better representations. GNNs have achieved state-of-the-art results on collaborative filtering tasks.

## Topics

### [1. Graph-Based Recommendation](graph-basics.md)
- User-item bipartite graph
- Graph collaborative signal
- Random walks, PageRank
- Metapath approaches

### [2. GNNs for RecSys](gnn-fundamentals.md)
- Message passing framework
- Graph Convolutional Networks (GCN)
- GraphSAGE
- **PinSage**: Pinterest's GNN at scale

**Paper**: Ying et al. (2018). "Graph convolutional neural networks for web-scale recommender systems". *KDD*.

### [3. LightGCN](lightgcn.md)
**Simplified GNN** that outperforms complex architectures

**Key Insight**: Feature transformation and non-linearity unnecessary for CF

**Architecture**:
- Only neighborhood aggregation
- Linear propagation
- Layer combination

**Results**: 16% improvement over NGCF

**Paper**: He et al. (2020). "LightGCN: Simplifying and powering graph convolution network". *SIGIR*.

### [4. Knowledge Graph Integration](knowledge-graphs.md)
- Knowledge graph embeddings
- **KGAT**: Knowledge Graph Attention Networks
- Multi-task learning with KG

## Performance

| Model | MovieLens (Recall@20) | Amazon-Book (Recall@20) |
|-------|----------------------|------------------------|
| MF | 0.1698 | 0.0315 |
| NGCF | 0.1844 | 0.0344 |
| **LightGCN** | **0.2131** | **0.0411** |

*Return to [Main Course Page](../README.md)*
