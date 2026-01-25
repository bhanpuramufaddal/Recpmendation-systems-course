# Week 6: Sequential and Session-Based Recommendations

## Overview

Sequential recommendation models the temporal dynamics of user behavior, predicting what users will interact with next based on their interaction history. This is crucial for session-based platforms like e-commerce and video streaming.

## Topics

### [1. Modeling User Sequences](modeling-sequences.md)
- Markov chains
- Session-based vs. long-term modeling
- Next-item prediction

### [2. RNNs for Recommendations](rnn-recsys.md)
- **GRU4Rec**: Session-based with GRUs
- LSTM for user behavior
- Attention mechanisms

**Paper**: Hidasi et al. (2016). "Session-based recommendations with recurrent neural networks". *ICLR*.

### [3. Transformer Architectures](transformers.md)
- **BERT4Rec**: Bidirectional with masked prediction
- **SASRec**: Self-attentive sequential recommendation
- Positional encodings

**Papers**:
- Sun et al. (2019). "BERT4Rec". *CIKM*.
- Kang & McAuley (2018). "SASRec". *ICDM*.

### [4. Interactive Recommendations](interactive.md)
- Conversational systems
- Reinforcement learning formulation
- Contextual bandits

## Key Results

**GRU4Rec** (RSC15 dataset):
- 15-20% improvement over item-kNN
- Training: ~4 hours on GPU

**BERT4Rec** (MovieLens 1M):
- 7.24% HR@10 improvement
- 11.46% MRR improvement

*Return to [Main Course Page](../README.md)*
