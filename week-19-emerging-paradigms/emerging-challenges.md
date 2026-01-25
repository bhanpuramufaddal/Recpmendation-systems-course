# Week 19: Emerging Challenges

## Overview

**New frontiers**: Challenges as recommendation systems evolve.

**Categories**:
1. **Ephemeral content**: TikTok, Reels, Stories
2. **Metaverse**: 3D virtual worlds
3. **Sustainability**: Carbon footprint of ML
4. **Real-time edge**: On-device recommendations
5. **Multimodal**: Integrating video, audio, text, images

---

## Ephemeral and Short-Form Content

### TikTok and Instagram Reels

**Challenges**:
- **High velocity**: Millions of videos uploaded daily
- **Short lifespan**: Viral for hours, then forgotten
- **Cold start**: New videos need immediate exposure
- **Attention span**: 15-60 second videos

**Key metrics**: Completion rate (did user watch full video?)

**Approach**:
- Real-time feature extraction (video, audio, captions)
- Viral prediction models
- Rapid A/B testing (hours, not weeks)
- Creator diversity (avoid homogeneity)

---

## Metaverse Recommendations

### Virtual Worlds

**Platforms**: Meta Horizon, Roblox, VRChat, Decentraland.

**New dimensions**:
- **Spatial**: 3D location matters
- **Social**: Real-time co-presence
- **Avatar**: Virtual identity and customization
- **Immersive**: VR/AR interactions

**Recommendation challenges**:
- **3D scene understanding**: Recommend objects in virtual space
- **Social context**: Who else is in the room?
- **Cross-reality**: Physical → virtual preferences
- **Privacy**: Biometric data (gaze tracking, gestures)

---

## Sustainability and Green ML

### Carbon Footprint

**Problem**: Training large models emits significant CO2.

**Example**:
- GPT-3 training: ~500 tons CO2
- Recommendation model retraining: Daily/weekly

**Solutions**:
- **Efficient architectures**: Distillation, pruning, quantization
- **Transfer learning**: Reduce training from scratch
- **Green data centers**: Renewable energy
- **On-device inference**: Reduce server load

---

## Real-Time Edge Recommendations

### On-Device ML

**Motivation**:
- **Latency**: Sub-millisecond recommendations
- **Privacy**: Data stays on device
- **Offline**: Work without internet

**Challenges**:
- **Model size**: Limited to 10-100 MB
- **Compute**: Low-power CPUs/GPUs
- **Memory**: Limited RAM (1-2 GB)

**Solutions**:
- **Model compression**: Quantization (FP32 → INT8)
- **Knowledge distillation**: Large model → small model
- **On-device training**: Federated learning
- **Hybrid**: Critical features on-device, complex models in cloud

---

## Multimodal Integration

### Beyond Text and Images

**Modalities**:
- **Video**: Visual + audio + captions
- **Audio**: Music, podcasts, voice
- **Text**: Descriptions, reviews, queries
- **Images**: Product photos, thumbnails
- **Sensors**: Location, biometrics, activity

**Challenge**: Fuse modalities effectively.

**Approaches**:
- **Early fusion**: Concatenate embeddings
- **Late fusion**: Separate models, combine scores
- **Cross-modal**: Attention across modalities (CLIP-style)

---

## Summary

**Key Takeaways**:
1. **Ephemeral content**: High velocity, short lifespan (TikTok)
2. **Metaverse**: 3D spatial, social, immersive recommendations
3. **Sustainability**: Green ML, efficient architectures
4. **Edge computing**: On-device recommendations for privacy/latency
5. **Multimodal**: Video, audio, text, image fusion

**Future directions**:
- Real-time adaptation (sub-second)
- Privacy-preserving techniques (federated, differential privacy)
- Sustainable AI practices
- Cross-reality experiences (physical ↔ virtual)

---

## References

1. **Strubell, E., et al. (2019)**. "Energy and Policy Considerations for Deep Learning in NLP". *ACL*.

2. **Schwartz, R., et al. (2020)**. "Green AI". *CACM*.

3. **Wu, C., et al. (2020)**. "Machine Learning at Facebook: Understanding Inference at the Edge". *HPCA*.
