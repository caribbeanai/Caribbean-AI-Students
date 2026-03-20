"""
=============================================================================
Caribbean AI Academy - Sixth Form (Ages 16-18)
Lesson 05: Generative Models (GANs)
=============================================================================
Topic: Generative Adversarial Networks — Generating Synthetic Caribbean
       Economic Data

Wah gwaan! Today we tackling one of the most creative areas of AI —
Generative Models. Specifically, GANs (Generative Adversarial Networks).

A GAN has TWO neural networks competing against each other:
  - GENERATOR: Creates fake data (like a counterfeiter printing money)
  - DISCRIMINATOR: Tries to spot fake vs real data (like a bank teller)

They train together in a game — the Generator gets better at faking,
the Discriminator gets better at detecting. Eventually, the Generator
produces data so realistic yuh cyah tell it from the real ting!

Caribbean Application: We generate synthetic economic data for
Caribbean nations. Why? Real economic data is often scarce, incomplete,
or confidential. Synthetic data can fill gaps for research and planning.

Requirements: pip install torch numpy
Author: Adrian Dunkley | Caribbean AI Academy
=============================================================================
"""

import numpy as np
import random

print("=" * 65)
print("  LESSON 05: Generative Adversarial Networks (GANs)")
print("  Generating Synthetic Caribbean Economic Data")
print("=" * 65)


# =====================================================================
# SECTION 1: Understanding GANs
# =====================================================================
print("\n--- Part 1: How GANs Work ---\n")
print("""
Imagine a scene in a Caribbean marketplace:

  GENERATOR (The Artist):
  "I'm painting fake Jamaican dollar bills. At first they look
   terrible — wrong colour, wrong size. But I keep improving!"

  DISCRIMINATOR (The Inspector):
  "I'm checking every bill. Real or fake? At first it's easy to spot
   fakes. But the artist keeps getting better..."

  THE GAME:
  The Artist wants to fool the Inspector.
  The Inspector wants to catch the Artist.
  Both get better and better until the fakes are PERFECT.

  That's a GAN! Two networks in adversarial training.

Real Caribbean uses for GANs:
- Generate synthetic health data (protect patient privacy)
- Augment small agricultural datasets (dasheen leaf diseases)
- Create realistic hurricane scenarios for planning
- Generate synthetic economic data for policy modelling
""")


# =====================================================================
# SECTION 2: Generate Real Caribbean Economic Data
# =====================================================================
print("--- Part 2: Real (Synthetic) Caribbean Economic Data ---\n")

CARIBBEAN_COUNTRIES = [
    'Jamaica', 'Trinidad & Tobago', 'Barbados', 'Bahamas',
    'Guyana', 'Suriname', 'Belize', 'Haiti', 'Grenada',
    'St Lucia', 'Dominica', 'Antigua & Barbuda', 'St Kitts',
    'St Vincent', 'Cuba', 'Dominican Republic'
]


def generate_real_economic_data(n_samples=500):
    """
    Generate "real" Caribbean economic indicators.
    These follow realistic patterns based on actual Caribbean economies.

    Features (6 dimensions):
    1. GDP per capita (USD, thousands) — ranges from ~1k (Haiti) to ~30k (Bahamas)
    2. Tourism dependency (% of GDP) — Caribbean average ~30-50%
    3. Agricultural output (% of GDP) — varies widely
    4. Unemployment rate (%) — Caribbean averages 10-20%
    5. Debt-to-GDP ratio (%) — many Caribbean nations >60%
    6. Remittance inflow (% of GDP) — significant for Jamaica, Haiti, Guyana
    """
    data = []
    for _ in range(n_samples):
        # Sample from different Caribbean economy "archetypes"
        archetype = random.choice(['tourism', 'resource', 'mixed', 'developing'])

        if archetype == 'tourism':
            # Bahamas, Barbados, Antigua style
            gdp = np.random.normal(18, 5)
            tourism = np.random.normal(45, 10)
            agriculture = np.random.normal(3, 1.5)
            unemployment = np.random.normal(12, 4)
            debt = np.random.normal(70, 15)
            remittance = np.random.normal(3, 1.5)
        elif archetype == 'resource':
            # Trinidad, Guyana, Suriname style
            gdp = np.random.normal(15, 6)
            tourism = np.random.normal(10, 5)
            agriculture = np.random.normal(8, 3)
            unemployment = np.random.normal(8, 3)
            debt = np.random.normal(50, 15)
            remittance = np.random.normal(2, 1)
        elif archetype == 'mixed':
            # Jamaica, Dominican Republic style
            gdp = np.random.normal(8, 3)
            tourism = np.random.normal(25, 8)
            agriculture = np.random.normal(7, 3)
            unemployment = np.random.normal(15, 5)
            debt = np.random.normal(95, 20)
            remittance = np.random.normal(15, 5)
        else:
            # Haiti, less-developed economies
            gdp = np.random.normal(2, 1)
            tourism = np.random.normal(5, 3)
            agriculture = np.random.normal(20, 6)
            unemployment = np.random.normal(25, 8)
            debt = np.random.normal(40, 15)
            remittance = np.random.normal(25, 8)

        # Clip to reasonable ranges
        row = [
            max(gdp, 0.5), max(tourism, 0), max(agriculture, 0),
            max(unemployment, 1), max(debt, 10), max(remittance, 0)
        ]
        data.append(row)

    return np.array(data, dtype=np.float32)


FEATURE_NAMES = ['GDP/cap($K)', 'Tourism(%)', 'Agri(%)',
                 'Unemp(%)', 'Debt/GDP(%)', 'Remit(%)']

real_data = generate_real_economic_data(500)
print(f"Generated {len(real_data)} real economic data points")
print(f"Features: {FEATURE_NAMES}\n")
print("Real data statistics:")
for i, name in enumerate(FEATURE_NAMES):
    print(f"  {name:<14} mean={real_data[:, i].mean():.2f}  "
          f"std={real_data[:, i].std():.2f}  "
          f"range=[{real_data[:, i].min():.1f}, {real_data[:, i].max():.1f}]")


# =====================================================================
# SECTION 3: Build the GAN
# =====================================================================
print("\n--- Part 3: Building the GAN ---\n")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Showing conceptual architecture.")


if TORCH_AVAILABLE:
    # Normalize data to [-1, 1] range for GAN training
    data_min = real_data.min(axis=0)
    data_max = real_data.max(axis=0)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1
    normalized_data = 2 * (real_data - data_min) / data_range - 1
    real_tensor = torch.tensor(normalized_data)

    DATA_DIM = 6       # Number of economic features
    NOISE_DIM = 16      # Size of random noise input to Generator

    class Generator(nn.Module):
        """
        The Generator — the creative "artist" of the GAN.

        Takes random noise and transforms it into fake economic data.
        Like a Caribbean economist inventing realistic-looking statistics!
        (But for GOOD purposes — data augmentation and privacy!)
        """
        def __init__(self, noise_dim=16, output_dim=6):
            super(Generator, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(noise_dim, 32),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(32),
                nn.Linear(32, 64),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(64),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.2),
                nn.Linear(32, output_dim),
                nn.Tanh()  # Output in [-1, 1] range
            )

        def forward(self, noise):
            return self.network(noise)

    class Discriminator(nn.Module):
        """
        The Discriminator — the "inspector" of the GAN.

        Looks at economic data and decides: real or generated?
        Like a CARICOM auditor checking if economic reports are genuine!
        """
        def __init__(self, input_dim=6):
            super(Discriminator, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(32, 64),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Output probability: 0=fake, 1=real
            )

        def forward(self, data):
            return self.network(data)

    # Create models
    generator = Generator(NOISE_DIM, DATA_DIM)
    discriminator = Discriminator(DATA_DIM)

    print("Generator Architecture:")
    print(generator)
    print(f"\nDiscriminator Architecture:")
    print(discriminator)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\nGenerator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")


    # =================================================================
    # SECTION 4: Training the GAN
    # =================================================================
    print("\n--- Part 4: Training the GAN ---\n")

    # Loss and optimizers
    criterion = nn.BCELoss()  # Binary Cross Entropy
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    BATCH_SIZE = 64
    NUM_EPOCHS = 200

    # Labels
    real_label = 1.0
    fake_label = 0.0

    print(f"Training GAN for {NUM_EPOCHS} epochs...")
    print(f"Watch the Generator and Discriminator compete!\n")

    g_losses = []
    d_losses = []

    for epoch in range(NUM_EPOCHS):
        # --- Train Discriminator ---
        discriminator.train()
        d_optimizer.zero_grad()

        # Real data batch
        idx = np.random.choice(len(real_tensor), BATCH_SIZE, replace=False)
        real_batch = real_tensor[idx]
        real_labels = torch.full((BATCH_SIZE, 1), real_label)

        d_real_output = discriminator(real_batch)
        d_real_loss = criterion(d_real_output, real_labels)

        # Fake data from Generator
        noise = torch.randn(BATCH_SIZE, NOISE_DIM)
        fake_batch = generator(noise).detach()
        fake_labels = torch.full((BATCH_SIZE, 1), fake_label)

        d_fake_output = discriminator(fake_batch)
        d_fake_loss = criterion(d_fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # --- Train Generator ---
        generator.train()
        g_optimizer.zero_grad()

        noise = torch.randn(BATCH_SIZE, NOISE_DIM)
        fake_batch = generator(noise)
        # Generator wants Discriminator to think fakes are real
        g_output = discriminator(fake_batch)
        g_loss = criterion(g_output, real_labels)

        g_loss.backward()
        g_optimizer.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        if (epoch + 1) % 40 == 0:
            print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"D_loss: {d_loss.item():.4f}  "
                  f"G_loss: {g_loss.item():.4f}  "
                  f"D(real): {d_real_output.mean().item():.3f}  "
                  f"D(fake): {d_fake_output.mean().item():.3f}")


    # =================================================================
    # SECTION 5: Generate and Evaluate Synthetic Data
    # =================================================================
    print("\n--- Part 5: Evaluating Generated Data ---\n")

    generator.eval()
    with torch.no_grad():
        noise = torch.randn(200, NOISE_DIM)
        generated_normalized = generator(noise).numpy()

    # Denormalize back to original scale
    generated_data = (generated_normalized + 1) / 2 * data_range + data_min

    print("Comparison: Real vs Generated Economic Data\n")
    print(f"{'Feature':<14} {'Real Mean':>10} {'Gen Mean':>10} "
          f"{'Real Std':>10} {'Gen Std':>10}")
    print("-" * 56)
    for i, name in enumerate(FEATURE_NAMES):
        r_mean = real_data[:, i].mean()
        g_mean = generated_data[:, i].mean()
        r_std = real_data[:, i].std()
        g_std = generated_data[:, i].std()
        print(f"{name:<14} {r_mean:>10.2f} {g_mean:>10.2f} "
              f"{r_std:>10.2f} {g_std:>10.2f}")

    print("\nSample generated Caribbean economic profiles:")
    for i in range(5):
        country = random.choice(CARIBBEAN_COUNTRIES)
        row = generated_data[i]
        print(f"\n  [{country} - synthetic profile]")
        for j, name in enumerate(FEATURE_NAMES):
            print(f"    {name}: {row[j]:.1f}")

    # Quality score — simple correlation comparison
    real_corr = np.corrcoef(real_data.T)
    gen_corr = np.corrcoef(generated_data.T)
    corr_diff = np.abs(real_corr - gen_corr).mean()
    print(f"\nCorrelation structure similarity: {1 - corr_diff:.3f} "
          f"(1.0 = perfect match)")

else:
    print("""
    GAN Architecture (Conceptual):

    GENERATOR:
      Random Noise (16-dim) -> Linear(32) -> LeakyReLU -> BatchNorm
      -> Linear(64) -> LeakyReLU -> BatchNorm
      -> Linear(32) -> LeakyReLU -> Linear(6) -> Tanh
      Output: Fake economic data (6 features)

    DISCRIMINATOR:
      Economic Data (6-dim) -> Linear(32) -> LeakyReLU -> Dropout
      -> Linear(64) -> LeakyReLU -> Dropout
      -> Linear(32) -> LeakyReLU -> Linear(1) -> Sigmoid
      Output: Probability that data is real (0-1)
    """)


# =====================================================================
# SECTION 6: Caribbean Applications
# =====================================================================
print("\n--- Part 6: Caribbean Applications of Generative Models ---\n")
print("""
GANs and generative models have huge potential in the Caribbean:

1. SYNTHETIC HEALTH DATA (All Caribbean hospitals)
   - Generate realistic patient data without privacy concerns
   - Train medical AI when real data is limited or confidential
   - Critical for rare tropical diseases with few recorded cases

2. HURRICANE SCENARIO GENERATION (CDEMA, Caribbean Met Services)
   - Generate thousands of realistic hurricane scenarios
   - Test emergency response plans against synthetic but plausible storms
   - Like a virtual cricket net session — practice for every delivery!

3. ECONOMIC POLICY SIMULATION (Central Banks, CDB, CARICOM)
   - Generate synthetic economic scenarios for stress testing
   - What if oil prices drop? Tourism collapses? Climate disaster?
   - Like our demo, but at national planning scale

4. AGRICULTURAL DATA AUGMENTATION (UWI, CARDI)
   - Few images of Caribbean crop diseases? Generate more!
   - GANs create realistic training data for crop disease AI

5. CULTURAL CONTENT (Caribbean arts and media)
   - Generate Caribbean music patterns, visual art styles
   - AI-assisted creative tools for Caribbean artists
   - Preserve and propagate Caribbean cultural expressions

6. DRUG DISCOVERY (UWI Chemistry, Caribbean Pharma)
   - Generate novel molecular structures for tropical disease drugs
   - Caribbean plants already provide many medicines — AI accelerates
""")


# =====================================================================
# QUIZ
# =====================================================================
print("=" * 65)
print("  QUIZ: Generative Adversarial Networks")
print("=" * 65)
print("""
Q1: What are the two networks in a GAN?
    a) Encoder and Decoder
    b) Generator and Discriminator
    c) Input and Output
    d) Teacher and Student

Q2: What is the Generator's goal?
    a) To classify data correctly
    b) To create fake data realistic enough to fool the Discriminator
    c) To delete real data
    d) To compress data

Q3: What is the Discriminator's goal?
    a) To generate new data
    b) To correctly distinguish real data from generated (fake) data
    c) To merge two datasets
    d) To normalize data

Q4: Why do we use Tanh activation in the Generator's output?
    a) Because it sounds cool
    b) To produce output in the [-1, 1] range, matching our
       normalized data
    c) To make all outputs positive
    d) Because ReLU doesn't work in GANs

Q5: What is "mode collapse" in GAN training?
    a) When the computer crashes
    b) When the Generator produces only a few types of output instead
       of diverse samples — it found one trick to fool the Discriminator
       and keeps using it
    c) When the Discriminator gets too good
    d) When training finishes successfully

Q6: Why is synthetic economic data useful for Caribbean nations?
    a) It's cheaper than real data
    b) It can fill data gaps, protect privacy, enable policy
       simulations, and augment small datasets for AI training
    c) Because Caribbean data is always wrong
    d) To replace all real data collection

Q7: In our GAN, what does the noise vector represent?
    a) Static on a radio
    b) A random seed that the Generator transforms into structured
       data — each noise vector produces a different economic profile
    c) Errors in the data
    d) The real data in disguise

Q8: How can you tell if a GAN is training well?
    a) The Generator loss goes to zero
    b) Both Generator and Discriminator losses are balanced, and
       D(fake) approaches 0.5 (Discriminator can't tell real from fake)
    c) The Discriminator always wins
    d) Training takes a long time

Q9: A Caribbean health ministry has only 100 patient records for
    a rare tropical disease. How could a GAN help?
    a) Replace the 100 records with fake ones
    b) Generate additional synthetic patient records that match the
       statistical properties of the real data, expanding the dataset
       for AI training while preserving privacy
    c) Delete the 100 records
    d) GANs can't help with medical data

Q10: What ethical considerations arise when generating synthetic
     Caribbean data?
     a) None — synthetic data is always safe
     b) Potential for misuse (fake statistics), bias amplification
        if training data is biased, transparency about what is
        synthetic vs real, and ensuring generated data doesn't
        reinforce harmful stereotypes about Caribbean nations
     c) Only cost considerations
     d) Copyright issues only

(Answers in quiz_answers.md)
""")

print("=" * 65)
print("  Lesson 05 Complete! Yuh understand GANs now — creative AI!")
print("  Next up: Ethics and Bias in Caribbean AI")
print("=" * 65)
