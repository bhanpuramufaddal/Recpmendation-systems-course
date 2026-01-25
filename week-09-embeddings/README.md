# Week 9: Embeddings and Pre-training

## Overview

Embeddings are dense vector representations that capture semantic meaning. Pre-training strategies from NLP (Word2Vec, BERT) have been adapted for recommendations, and modern LLMs offer new possibilities.

## Topics

### [1. Learning Embeddings](embeddings-learning.md)
- **Item2Vec**: Word2Vec for items
- Skip-gram and CBOW adaptations
- Negative sampling
- Embedding quality evaluation

### [2. Pre-training Strategies](pretraining.md)
- Self-supervised learning
- Contrastive learning (SimCLR)
- Masked prediction tasks
- Cross-domain transfer

### [3. LLMs for Recommendations](llms-recsys.md)
**2024 State-of-the-Art**

**Approaches**:
- LLMs as feature extractors
- Prompt engineering for recommendation
- Zero-shot and few-shot
- Fine-tuning on recommendation data

**Models**: GPT-4, Claude, Llama, Gemini

### [4. Multi-Modal Recommendations](multimodal.md)
- Vision-language models
- **CLIP** embeddings
- Fusing text, image, audio signals
- Cross-modal retrieval

**Applications**:
- E-commerce (product images + descriptions)
- Fashion recommendations
- Video content understanding

## CLIP for Recommendations

**Architecture**:
```
Image Encoder (Vision Transformer) ─┐
                                     ├─> Contrastive Loss
Text Encoder (Transformer) ─────────┘
```

**Use Case**: Match product images to text queries

*Return to [Main Course Page](../README.md)*
