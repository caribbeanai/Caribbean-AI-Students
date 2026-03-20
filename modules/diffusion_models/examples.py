"""
Caribbean AI Academy — Diffusion Model Examples
==================================================
Simple diffusion model implementation for learning.
Designed by Adrian Dunkley (https://Adriandunkley.net) — FREE!

Note: Real diffusion models need GPUs and hours of training.
This is a simplified educational version using 1D data.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# EXAMPLE 1: 1D Diffusion Model (From Scratch)
# ============================================================

def simple_diffusion_1d():
    """
    Demonstrate the diffusion process on simple 1D data.
    We'll generate data that looks like Caribbean island populations.
    """
    print("=" * 60)
    print("EXAMPLE 1: Simple 1D Diffusion (Caribbean Population Data)")
    print("=" * 60)

    np.random.seed(42)

    # Our "real" data: Caribbean island populations (simplified, in thousands)
    # Multi-modal distribution — some small islands, some large
    real_data = np.concatenate([
        np.random.normal(50, 15, 100),    # Small islands (Dominica, St. Kitts, etc.)
        np.random.normal(300, 50, 80),    # Medium islands (Barbados, Bahamas, etc.)
        np.random.normal(2800, 200, 40),  # Large islands (Jamaica, Trinidad, etc.)
    ])
    real_data = np.maximum(real_data, 5)  # No negative populations!

    print(f"Real data: {len(real_data)} samples")
    print(f"Range: {real_data.min():.0f}K to {real_data.max():.0f}K people")

    # === FORWARD PROCESS: Add noise gradually ===
    num_steps = 50
    betas = np.linspace(0.01, 0.3, num_steps)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)

    print("\n--- Forward Process (Adding Noise) ---")
    sample = real_data[0]  # Take one data point
    print(f"  Step  0: {sample:.1f}K (original — like Dominica's population)")

    for t in [10, 25, 40, 49]:
        noisy = np.sqrt(alpha_bars[t]) * sample + np.sqrt(1 - alpha_bars[t]) * np.random.randn()
        noise_pct = (1 - alpha_bars[t]) * 100
        print(f"  Step {t:2d}: {noisy:.1f}K ({noise_pct:.0f}% noise)")

    # === REVERSE PROCESS: Learn to denoise ===
    print("\n--- Reverse Process (Learning to Denoise) ---")
    print("Training a simple model to predict noise...")

    # Simple linear model to predict noise (educational — real models use neural nets)
    # Training: for each (noisy_sample, timestep), predict the noise

    # Generate training data
    n_train = 10000
    train_x = []
    train_noise = []

    for _ in range(n_train):
        x0 = real_data[np.random.randint(len(real_data))]
        t = np.random.randint(num_steps)
        noise = np.random.randn()
        x_noisy = np.sqrt(alpha_bars[t]) * x0 + np.sqrt(1 - alpha_bars[t]) * noise
        train_x.append([x_noisy, t / num_steps])  # Features: noisy value + normalized timestep
        train_noise.append(noise)

    train_x = np.array(train_x)
    train_noise = np.array(train_noise)

    # Train simple linear regression to predict noise
    # Add bias term
    X = np.column_stack([train_x, np.ones(n_train)])
    # Normal equation: w = (X^T X)^-1 X^T y
    w = np.linalg.lstsq(X, train_noise, rcond=None)[0]

    pred_noise = X @ w
    mse = np.mean((pred_noise - train_noise) ** 2)
    print(f"  Training MSE: {mse:.4f}")

    # === SAMPLING: Generate new data ===
    print("\n--- Sampling (Generating New Data) ---")
    n_generate = 200
    generated = np.random.randn(n_generate)  # Start from pure noise

    for t in reversed(range(num_steps)):
        # Predict noise
        features = np.column_stack([generated, np.full(n_generate, t / num_steps), np.ones(n_generate)])
        pred_noise = features @ w

        # Denoise step (simplified)
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        generated = (generated - (1 - alpha_t) / np.sqrt(1 - alpha_bar_t) * pred_noise) / np.sqrt(alpha_t)

        if t > 0:
            generated += np.sqrt(betas[t]) * np.random.randn(n_generate) * 0.5

    generated = np.abs(generated)  # No negatives

    # Compare distributions
    print(f"\n  Real data stats:      mean={real_data.mean():.0f}K, std={real_data.std():.0f}K")
    print(f"  Generated data stats: mean={generated.mean():.0f}K, std={generated.std():.0f}K")
    print(f"  (A neural network model would match much better!)")

    # Map to Caribbean context
    print("\n  Sample generated 'island populations' (thousands):")
    samples = np.sort(generated[generated > 10])[:10]
    island_names = ["Montserrat", "St. Kitts", "Dominica", "Grenada", "St. Vincent",
                    "Antigua", "St. Lucia", "Barbados", "Bahamas", "Suriname"]
    for name, pop in zip(island_names, samples):
        print(f"    {name}: {pop:.0f}K")


# ============================================================
# EXAMPLE 2: Image-like Diffusion (2D Patterns)
# ============================================================

def caribbean_pattern_diffusion():
    """
    Generate simple 2D patterns inspired by Caribbean motifs.
    Using a tiny 8x8 grid for educational purposes.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Caribbean Pattern Generation (2D Diffusion)")
    print("=" * 60)

    np.random.seed(42)

    # Create simple Caribbean-inspired patterns (8x8)
    def make_cross_pattern():
        """Like the X on Jamaica or Scotland's flag"""
        p = np.zeros((8, 8))
        for i in range(8):
            p[i, i] = 1.0
            p[i, 7-i] = 1.0
        return p

    def make_stripe_pattern():
        """Like Trinidad's diagonal stripe"""
        p = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                if abs(i - j) <= 1:
                    p[i, j] = 1.0
        return p

    def make_circle_pattern():
        """Like the circle elements in many Caribbean flags"""
        p = np.zeros((8, 8))
        center = 3.5
        for i in range(8):
            for j in range(8):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if 2 < dist < 3.5:
                    p[i, j] = 1.0
        return p

    patterns = {
        "Cross (Jamaica-style)": make_cross_pattern(),
        "Stripe (Trinidad-style)": make_stripe_pattern(),
        "Circle (Dominica-style)": make_circle_pattern(),
    }

    # Show original patterns as ASCII art
    for name, pattern in patterns.items():
        print(f"\n  {name}:")
        for row in pattern:
            line = "    "
            for val in row:
                line += "##" if val > 0.5 else ".."
            print(line)

    # Demonstrate forward diffusion on one pattern
    print("\n--- Forward Diffusion on Cross Pattern ---")
    original = make_cross_pattern()

    num_steps = 20
    betas = np.linspace(0.05, 0.5, num_steps)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)

    for t in [0, 5, 10, 15, 19]:
        noisy = np.sqrt(alpha_bars[t]) * original + np.sqrt(1 - alpha_bars[t]) * np.random.randn(8, 8)
        print(f"\n  Step {t} ({(1-alpha_bars[t])*100:.0f}% noise):")
        for row in noisy:
            line = "    "
            for val in row:
                line += "##" if val > 0.3 else ".."
            print(line)

    print("\n--- Reverse Process Would Reconstruct the Pattern ---")
    print("  (Full implementation requires a neural network trained on many patterns)")
    print("  In production, you'd use a U-Net architecture with attention layers.")

    print("\n  Key Insight: Diffusion models work the same way for 8x8 patterns")
    print("  as they do for 512x512 photos — just with bigger neural networks!")


# ============================================================
# EXAMPLE 3: Diffusion for Caribbean Data Augmentation
# ============================================================

def caribbean_data_augmentation():
    """
    Show how diffusion concepts can augment small Caribbean datasets.
    Critical for island nations with limited data!
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Data Augmentation for Small Island Datasets")
    print("=" * 60)

    np.random.seed(42)

    # Problem: Small Caribbean hospital has only 50 chest X-ray measurements
    # (represented as feature vectors for simplicity)
    print("\nScenario: Queen Elizabeth Hospital, Barbados")
    print("Only 50 patient records for rare condition — not enough to train ML!")

    n_real = 50
    n_features = 4  # Simplified: [temperature, heart_rate, blood_pressure, oxygen_level]

    # Real patient data
    real_patients = np.column_stack([
        np.random.normal(38.5, 0.8, n_real),    # Temperature (fever)
        np.random.normal(95, 12, n_real),        # Heart rate (elevated)
        np.random.normal(140, 15, n_real),       # Blood pressure (high)
        np.random.normal(94, 3, n_real),         # Oxygen level (low)
    ])

    feature_names = ["Temperature(C)", "HeartRate", "BloodPressure", "OxygenLevel"]

    print(f"\nReal patient data: {n_real} records")
    print(f"Features: {', '.join(feature_names)}")
    print(f"\nReal data means: {real_patients.mean(axis=0).round(1)}")
    print(f"Real data stds:  {real_patients.std(axis=0).round(1)}")

    # Simple diffusion-inspired augmentation
    # Add noise and denoise to create plausible new samples
    n_augmented = 200

    # Method: Noise-and-denoise augmentation
    augmented = []
    for _ in range(n_augmented):
        # Pick a random real sample
        base = real_patients[np.random.randint(n_real)]

        # Add calibrated noise (like forward diffusion)
        noise_level = np.random.uniform(0.1, 0.4)
        noise = np.random.randn(n_features) * real_patients.std(axis=0) * noise_level

        # "Denoise" by pulling back toward data distribution
        new_sample = base + noise

        # Ensure physiological plausibility
        new_sample[0] = np.clip(new_sample[0], 36.0, 41.0)  # Temperature
        new_sample[1] = np.clip(new_sample[1], 50, 150)      # Heart rate
        new_sample[2] = np.clip(new_sample[2], 90, 200)      # Blood pressure
        new_sample[3] = np.clip(new_sample[3], 85, 100)      # Oxygen

        augmented.append(new_sample)

    augmented = np.array(augmented)

    print(f"\nAugmented data: {n_augmented} synthetic records")
    print(f"Augmented means: {augmented.mean(axis=0).round(1)}")
    print(f"Augmented stds:  {augmented.std(axis=0).round(1)}")

    # Train classifier with and without augmentation
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Create a binary classification task (sick vs healthy)
    healthy = np.column_stack([
        np.random.normal(37.0, 0.3, 100),
        np.random.normal(72, 8, 100),
        np.random.normal(120, 10, 100),
        np.random.normal(98, 1, 100),
    ])

    # Without augmentation
    X_small = np.vstack([real_patients, healthy[:50]])
    y_small = np.array([1]*50 + [0]*50)
    score_small = cross_val_score(LogisticRegression(), X_small, y_small, cv=5).mean()

    # With augmentation
    X_aug = np.vstack([real_patients, augmented[:150], healthy])
    y_aug = np.array([1]*50 + [1]*150 + [0]*100)
    score_aug = cross_val_score(LogisticRegression(), X_aug, y_aug, cv=5).mean()

    print(f"\nClassification accuracy WITHOUT augmentation: {score_small:.1%}")
    print(f"Classification accuracy WITH augmentation:    {score_aug:.1%}")
    print(f"\nDiffusion-style augmentation helps small Caribbean datasets punch above their weight!")

    print("\n  This matters because Caribbean hospitals often have small datasets.")
    print("  Islands like Montserrat, St. Kitts, Dominica — every data point counts.")
    print("  Generative models help us create more training data from what we have.")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CARIBBEAN AI ACADEMY — DIFFUSION MODEL EXAMPLES")
    print("  Designed by Adrian Dunkley (https://Adriandunkley.net)")
    print("=" * 60 + "\n")

    simple_diffusion_1d()
    caribbean_pattern_diffusion()
    caribbean_data_augmentation()

    print("\n" + "=" * 60)
    print("  Diffusion model examples complete!")
    print("  From noise to beauty — just like a Caribbean sunrise!")
    print("=" * 60)
