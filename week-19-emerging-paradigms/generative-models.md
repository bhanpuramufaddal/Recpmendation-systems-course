# Week 19: Generative Recommendations

## Overview

**Generative models**: Generate new items or synthetic data for recommendations.

**Applications**:
1. **Item generation**: Create personalized items (music, images, text)
2. **Data augmentation**: Synthesize training data
3. **Counterfactual generation**: "What if" scenarios for evaluation
4. **Cold-start**: Generate features for new items

**Key techniques**:
- Variational Autoencoders (VAE)
- Generative Adversarial Networks (GAN)
- Diffusion Models
- Large Language Models (LLM)

---

## Variational Autoencoders for CF

### VAE-CF

**Idea**: Learn latent representation of user preferences using VAE.

**Architecture**:
1. **Encoder**: User interaction history → latent code $z$
2. **Decoder**: Latent code $z$ → predicted preferences

**Loss**: Reconstruction + KL divergence

$$\mathcal{L} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + \text{KL}(q(z|x) \| p(z))$$

---

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_CF(nn.Module):
    def __init__(self, n_items, latent_dim=64):
        super().__init__()

        # Encoder
        self.encoder_fc1 = nn.Linear(n_items, 600)
        self.encoder_fc2_mean = nn.Linear(600, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(600, latent_dim)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, 600)
        self.decoder_fc2 = nn.Linear(600, n_items)

    def encode(self, x):
        """
        Encode user interaction history to latent distribution.

        Args:
            x: [batch_size, n_items] binary interaction matrix

        Returns:
            mean, logvar: Parameters of latent distribution
        """
        h = torch.tanh(self.encoder_fc1(x))
        mean = self.encoder_fc2_mean(h)
        logvar = self.encoder_fc2_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """Reparameterization trick for backprop through sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        """
        Decode latent code to item preferences.

        Args:
            z: [batch_size, latent_dim]

        Returns:
            logits: [batch_size, n_items]
        """
        h = torch.tanh(self.decoder_fc1(z))
        logits = self.decoder_fc2(h)
        return logits

    def forward(self, x):
        """
        Forward pass through VAE.

        Args:
            x: [batch_size, n_items] interaction matrix

        Returns:
            recon_logits: Reconstructed interaction logits
            mean, logvar: Latent distribution parameters
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mean, logvar


def vae_loss(recon_logits, x, mean, logvar, beta=0.2):
    """
    VAE loss function.

    Args:
        recon_logits: Reconstructed logits
        x: Original interaction matrix
        mean, logvar: Latent distribution parameters
        beta: KL divergence weight

    Returns:
        loss: Total loss
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = -torch.sum(
        F.log_softmax(recon_logits, dim=1) * x,
        dim=1
    ).mean()

    # KL divergence
    kl_div = -0.5 * torch.sum(
        1 + logvar - mean.pow(2) - logvar.exp(),
        dim=1
    ).mean()

    # Total loss
    loss = recon_loss + beta * kl_div

    return loss, recon_loss, kl_div


# Training
model = VAE_CF(n_items=10000, latent_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for user_interactions in train_loader:
        # user_interactions: [batch_size, n_items] binary matrix

        # Forward pass
        recon_logits, mean, logvar = model(user_interactions)

        # Compute loss
        loss, recon_loss, kl_div = vae_loss(recon_logits, user_interactions, mean, logvar)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Recon = {recon_loss:.4f}, KL = {kl_div:.4f}")


# Recommendation
def recommend(model, user_interaction, k=10):
    """
    Generate recommendations for user.

    Args:
        user_interaction: [n_items] binary vector
        k: Number of recommendations

    Returns:
        top_k_items: Recommended item indices
    """
    model.eval()

    with torch.no_grad():
        # Encode user
        mean, logvar = model.encode(user_interaction.unsqueeze(0))

        # Sample latent code
        z = model.reparameterize(mean, logvar)

        # Decode to item scores
        logits = model.decode(z).squeeze()

        # Mask already-interacted items
        logits[user_interaction == 1] = -float('inf')

        # Top-K
        top_k_items = torch.topk(logits, k).indices

    return top_k_items
```

---

## GANs for Data Augmentation

### Motivation

**Problem**: Limited user-item interactions (sparsity).

**Solution**: Generate synthetic interactions to augment training data.

**GAN approach**:
- **Generator**: Creates fake user interactions
- **Discriminator**: Distinguishes real vs. fake
- **Training**: Generator learns to fool discriminator

---

### Implementation

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=64, n_items=10000):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, n_items)

    def forward(self, z):
        """
        Generate fake user interactions.

        Args:
            z: [batch_size, latent_dim] random noise

        Returns:
            fake_interactions: [batch_size, n_items]
        """
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output in [0, 1]
        return x


class Discriminator(nn.Module):
    def __init__(self, n_items=10000):
        super().__init__()
        self.fc1 = nn.Linear(n_items, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Discriminate real vs. fake interactions.

        Args:
            x: [batch_size, n_items] interaction matrix

        Returns:
            logits: [batch_size] real/fake logits
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits.squeeze()


# Training
generator = Generator(latent_dim=64, n_items=10000)
discriminator = Discriminator(n_items=10000)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

criterion = nn.BCEWithLogitsLoss()

for epoch in range(100):
    for real_interactions in train_loader:
        batch_size = real_interactions.size(0)

        # Labels
        real_labels = torch.ones(batch_size)
        fake_labels = torch.zeros(batch_size)

        # ========== Train Discriminator ==========
        d_optimizer.zero_grad()

        # Real interactions
        d_real = discriminator(real_interactions)
        d_loss_real = criterion(d_real, real_labels)

        # Fake interactions
        z = torch.randn(batch_size, 64)
        fake_interactions = generator(z)
        d_fake = discriminator(fake_interactions.detach())
        d_loss_fake = criterion(d_fake, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # ========== Train Generator ==========
        g_optimizer.zero_grad()

        # Generate fake interactions
        z = torch.randn(batch_size, 64)
        fake_interactions = generator(z)

        # Fool discriminator
        d_fake = discriminator(fake_interactions)
        g_loss = criterion(d_fake, real_labels)  # Want discriminator to think they're real

        g_loss.backward()
        g_optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: D_loss = {d_loss:.4f}, G_loss = {g_loss:.4f}")


# Data augmentation
def augment_training_data(generator, n_synthetic=1000):
    """
    Generate synthetic user interactions.

    Args:
        n_synthetic: Number of synthetic users to generate

    Returns:
        synthetic_interactions: [n_synthetic, n_items]
    """
    generator.eval()

    with torch.no_grad():
        z = torch.randn(n_synthetic, 64)
        synthetic_interactions = generator(z)

        # Binarize (threshold at 0.5)
        synthetic_interactions = (synthetic_interactions > 0.5).float()

    return synthetic_interactions
```

---

## Diffusion Models for Recommendations

### Denoising Diffusion

**Idea**: Learn to denoise user preferences gradually.

**Process**:
1. **Forward**: Add noise to user interactions over T steps
2. **Reverse**: Train model to denoise (predict original from noisy)
3. **Generation**: Start from noise, iteratively denoise

**Application**: Generate personalized item sequences.

---

### Simplified Implementation

```python
class DiffusionRecommender(nn.Module):
    def __init__(self, n_items, hidden_dim=256, timesteps=100):
        super().__init__()
        self.n_items = n_items
        self.timesteps = timesteps

        # Noise schedule (variance at each timestep)
        self.betas = torch.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Denoising network
        self.denoiser = nn.Sequential(
            nn.Linear(n_items + 1, hidden_dim),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_items)
        )

    def add_noise(self, x, t):
        """
        Add noise to interactions at timestep t.

        Args:
            x: [batch_size, n_items] clean interactions
            t: [batch_size] timesteps

        Returns:
            x_t: Noisy interactions
            noise: Added noise
        """
        noise = torch.randn_like(x)

        # Get alpha for timestep
        alpha_t = self.alphas_cumprod[t].view(-1, 1)

        # Forward process: x_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * noise
        x_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

        return x_t, noise

    def denoise(self, x_t, t):
        """
        Predict noise at timestep t.

        Args:
            x_t: [batch_size, n_items] noisy interactions
            t: [batch_size] timesteps

        Returns:
            predicted_noise: [batch_size, n_items]
        """
        # Concatenate timestep as feature
        t_normalized = t.float() / self.timesteps
        input = torch.cat([x_t, t_normalized.unsqueeze(1)], dim=1)

        predicted_noise = self.denoiser(input)
        return predicted_noise

    def forward(self, x):
        """
        Training forward pass.

        Args:
            x: [batch_size, n_items] clean interactions

        Returns:
            loss: Denoising loss
        """
        batch_size = x.size(0)

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,))

        # Add noise
        x_t, noise = self.add_noise(x, t)

        # Predict noise
        predicted_noise = self.denoise(x_t, t)

        # Loss: MSE between predicted and actual noise
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, batch_size=1):
        """
        Generate synthetic user interactions via reverse diffusion.

        Returns:
            generated: [batch_size, n_items]
        """
        # Start from pure noise
        x = torch.randn(batch_size, self.n_items)

        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            # Predict noise
            t_batch = torch.full((batch_size,), t, dtype=torch.long)
            predicted_noise = self.denoise(x, t_batch)

            # Remove predicted noise
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

            x = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

            # Add noise for next step (except last step)
            if t > 0:
                noise = torch.randn_like(x)
                x = x * torch.sqrt(alpha_t_prev) + noise * torch.sqrt(1 - alpha_t_prev)

        # Binarize
        generated = (x > 0).float()

        return generated


# Training
model = DiffusionRecommender(n_items=10000, hidden_dim=256, timesteps=100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for user_interactions in train_loader:
        loss = model(user_interactions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Generate synthetic users
synthetic_users = model.sample(batch_size=100)
print(f"Generated {len(synthetic_users)} synthetic users")
```

---

## Counterfactual Generation

### Motivation

**Problem**: Can't observe all user-item interactions (missing data).

**Solution**: Generate counterfactual interactions ("What if user saw item X?").

**Use case**: Unbiased evaluation, debiasing models.

---

### Implementation

```python
def generate_counterfactuals(model, user, observed_items, candidate_items):
    """
    Generate counterfactual interactions.

    Args:
        model: Generative model (VAE or GAN)
        user: User ID
        observed_items: Items user actually interacted with
        candidate_items: Items to generate counterfactuals for

    Returns:
        counterfactual_prefs: Predicted preferences for candidate items
    """
    # Encode user from observed interactions
    user_vector = create_interaction_vector(observed_items, n_items=10000)

    # Encode to latent space
    with torch.no_grad():
        mean, logvar = model.encode(user_vector.unsqueeze(0))
        z = model.reparameterize(mean, logvar)

    # Decode to get full preference vector
    with torch.no_grad():
        full_prefs = torch.sigmoid(model.decode(z)).squeeze()

    # Extract counterfactual preferences for candidate items
    counterfactual_prefs = full_prefs[candidate_items]

    return counterfactual_prefs


# Example: Evaluate model with counterfactuals
observed_items = torch.tensor([10, 25, 47])
candidate_items = torch.tensor([100, 200, 300])  # Unobserved items

counterfactual_prefs = generate_counterfactuals(vae_model, user_id, observed_items, candidate_items)

print(f"Counterfactual preferences: {counterfactual_prefs}")
# Output: tensor([0.82, 0.15, 0.63])
```

---

## Summary

**Key Takeaways**:
1. **VAE-CF**: Probabilistic latent representation for collaborative filtering
2. **GANs**: Generate synthetic user interactions for data augmentation
3. **Diffusion**: Iterative denoising for high-quality generation
4. **Counterfactuals**: Fill in missing interactions for unbiased evaluation

**Applications**:
- Cold-start (generate features)
- Data augmentation (alleviate sparsity)
- Counterfactual reasoning (unbiased evaluation)
- Personalized content generation

**Best practices**:
- VAE for interpretable latent space
- GAN for high-quality synthetic data
- Diffusion for fine-grained control over generation

---

## Practice Problems

**Problem 1**: Train VAE-CF on MovieLens. Visualize latent space (t-SNE). Do similar users cluster together?

**Problem 2**: Use GAN to augment sparse users (< 5 interactions). Does it improve cold-start performance?

**Problem 3**: Implement diffusion model for sequential recommendation (generate item sequences instead of sets).

**Problem 4**: Generate counterfactual interactions for unbiased evaluation. Compare NDCG on observed vs. counterfactual data.

---

## References

1. **Liang, D., et al. (2018)**. "Variational Autoencoders for Collaborative Filtering". *WWW*.

2. **Goodfellow, I., et al. (2014)**. "Generative Adversarial Networks". *NeurIPS*.

3. **Ho, J., et al. (2020)**. "Denoising Diffusion Probabilistic Models". *NeurIPS*.

4. **Wang, X., et al. (2021)**. "Counterfactual Data Augmentation for Neural Machine Translation". *NAACL*.
