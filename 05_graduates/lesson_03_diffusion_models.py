"""
=============================================================================
 CARIBBEAN AI ACADEMY - GRADUATES MODULE (05)
 Lesson 03: Diffusion Models (DDPM) from Scratch
 By Adrian Dunkley
=============================================================================

 Big tings, Graduates! Today we tackle Diffusion Models -
 di same technology behind DALL-E, Midjourney, and Stable Diffusion.

 Imagine generating Caribbean art, fashion designs, or even
 synthetic satellite imagery fi hurricane prediction -
 all from noise. Dat is di power of diffusion.

 We implement Denoising Diffusion Probabilistic Models (DDPM)
 using numpy to truly understand di mathematics.

 LEARNING OBJECTIVES:
 1. Understand forward diffusion (adding noise)
 2. Implement reverse diffusion (denoising)
 3. Build noise schedule (linear beta schedule)
 4. Apply to Caribbean-themed data patterns
 5. Grasp the connection to score matching and SDEs

 # 🧩 Puzzle Piece 9/12: The animal is known for its color... not blue, not red, but ___
=============================================================================
"""

import numpy as np
from typing import Tuple, List, Dict

# =============================================================================
# PART 1: NOISE SCHEDULE - Di Heartbeat of Diffusion
# =============================================================================

class NoiseSchedule:
    """
    Beta schedule controls how much noise we add at each timestep.

    Think of it like a hurricane building:
    - t=0: Clear sky (original data, no noise)
    - t=T/2: Storm building (half noise, half signal)
    - t=T: Full hurricane (pure noise, no signal left)

    Di reverse process is like di storm clearing -
    we gradually remove noise to reveal di original pattern.
    """

    def __init__(self, num_timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = num_timesteps

        # Linear beta schedule
        self.betas = np.linspace(beta_start, beta_end, num_timesteps)

        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        self.alpha_cumprod_prev = np.concatenate([[1.0], self.alpha_cumprod[:-1]])

        # For q(x_t | x_0) - di forward process shortcut
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - self.alpha_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

    def show_schedule(self):
        """Visualize di noise schedule."""
        print("=" * 60)
        print(" NOISE SCHEDULE VISUALIZATION")
        print("=" * 60)
        checkpoints = [0, 100, 250, 500, 750, 999]
        print(f"\n{'Timestep':>10} {'Beta':>10} {'Alpha_bar':>12} {'Signal%':>10} {'Noise%':>10}")
        print("-" * 55)
        for t in checkpoints:
            if t < self.T:
                signal = self.sqrt_alpha_cumprod[t] * 100
                noise = self.sqrt_one_minus_alpha_cumprod[t] * 100
                print(f"{t:>10} {self.betas[t]:>10.6f} {self.alpha_cumprod[t]:>12.6f} "
                      f"{signal:>9.1f}% {noise:>9.1f}%")


# =============================================================================
# PART 2: FORWARD PROCESS - Adding Noise (Di Storm Building)
# =============================================================================

class ForwardDiffusion:
    """
    Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    Key insight: We can jump directly to ANY timestep t from x_0!
    No need fi step through every intermediate step.
    Dis is because di sum of Gaussians is still Gaussian.
    """

    def __init__(self, schedule: NoiseSchedule):
        self.schedule = schedule

    def add_noise(self, x_0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise to clean data x_0 at timestep t.

        Returns: (noisy_data, noise_that_was_added)
        """
        noise = np.random.randn(*x_0.shape)

        sqrt_alpha_bar = self.schedule.sqrt_alpha_cumprod[t]
        sqrt_one_minus_alpha_bar = self.schedule.sqrt_one_minus_alpha_cumprod[t]

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def demonstrate_forward_process(self, x_0: np.ndarray):
        """Show how data gets progressively noisier."""
        print("\n" + "=" * 60)
        print(" FORWARD DIFFUSION PROCESS")
        print("=" * 60)
        print(f"\nOriginal data stats: mean={x_0.mean():.3f}, std={x_0.std():.3f}")

        timesteps = [0, 50, 100, 250, 500, 750, 999]
        for t in timesteps:
            x_t, noise = self.add_noise(x_0, t)
            snr = self.schedule.alpha_cumprod[t] / (1 - self.schedule.alpha_cumprod[t])
            print(f"  t={t:>4}: mean={x_t.mean():>7.3f}, std={x_t.std():>6.3f}, "
                  f"SNR={snr:>8.3f}")


# =============================================================================
# PART 3: SIMPLE DENOISER - Learning to Remove Noise
# =============================================================================

class SimpleDenoiser:
    """
    A simple neural network (simulated) dat learns to predict noise.

    In real DDPM, dis would be a U-Net with attention.
    Here we use a simple MLP to show di concept.

    Di network learns: given noisy data x_t and timestep t,
    predict di noise epsilon dat was added.
    """

    def __init__(self, data_dim: int, hidden_dim: int = 64):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        # Simple 2-layer MLP weights
        np.random.seed(42)
        self.W1 = np.random.randn(data_dim + 1, hidden_dim) * 0.1  # +1 for timestep
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, data_dim) * 0.1
        self.b3 = np.zeros(data_dim)

    def predict_noise(self, x_t: np.ndarray, t: int, T: int = 1000) -> np.ndarray:
        """
        Predict the noise in x_t.

        In real implementation, di network would be trained via:
        Loss = ||epsilon - epsilon_theta(x_t, t)||^2

        Where epsilon is di actual noise added, and epsilon_theta
        is what di network predicts.
        """
        # Normalize timestep to [0, 1]
        t_normalized = np.array([t / T])

        # Concatenate data with timestep embedding
        if x_t.ndim == 1:
            input_vec = np.concatenate([x_t, t_normalized])
        else:
            t_broadcast = np.full((x_t.shape[0], 1), t / T)
            input_vec = np.concatenate([x_t, t_broadcast], axis=1)

        # Forward pass through MLP
        h1 = np.maximum(0, input_vec @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        predicted_noise = h2 @ self.W3 + self.b3

        return predicted_noise

    def training_step(self, x_0: np.ndarray, schedule: NoiseSchedule,
                      learning_rate: float = 0.001) -> float:
        """
        One training step of DDPM.

        Algorithm:
        1. Sample random timestep t ~ Uniform(0, T)
        2. Sample noise epsilon ~ N(0, I)
        3. Create noisy version: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
        4. Predict noise: epsilon_hat = model(x_t, t)
        5. Loss = MSE(epsilon, epsilon_hat)
        6. Update model weights
        """
        # Sample random timestep
        t = np.random.randint(0, schedule.T)

        # Sample noise
        noise = np.random.randn(*x_0.shape)

        # Create noisy data
        sqrt_ab = schedule.sqrt_alpha_cumprod[t]
        sqrt_1_ab = schedule.sqrt_one_minus_alpha_cumprod[t]
        x_t = sqrt_ab * x_0 + sqrt_1_ab * noise

        # Predict noise (forward pass)
        predicted_noise = self.predict_noise(x_t, t, schedule.T)

        # Compute MSE loss
        loss = np.mean((noise - predicted_noise) ** 2)

        return loss


# =============================================================================
# PART 4: REVERSE PROCESS - Generating from Noise (Di Storm Clearing)
# =============================================================================

class ReverseDiffusion:
    """
    Reverse process: generate data by iteratively denoising.

    Start from pure noise x_T ~ N(0, I)
    Iteratively denoise: x_{t-1} = f(x_t, t)

    p(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_t^2 I)

    Where mu_theta uses di predicted noise to estimate di mean.
    """

    def __init__(self, schedule: NoiseSchedule, denoiser: SimpleDenoiser):
        self.schedule = schedule
        self.denoiser = denoiser

    def reverse_step(self, x_t: np.ndarray, t: int) -> np.ndarray:
        """
        One step of di reverse process: x_{t-1} from x_t.

        mu = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta)
        x_{t-1} = mu + sigma_t * z   (where z ~ N(0,I) for t > 0)
        """
        alpha_t = self.schedule.alphas[t]
        beta_t = self.schedule.betas[t]
        alpha_bar_t = self.schedule.alpha_cumprod[t]

        # Predict noise
        predicted_noise = self.denoiser.predict_noise(x_t, t, self.schedule.T)

        # Compute mean
        coeff = beta_t / np.sqrt(1.0 - alpha_bar_t)
        mu = (1.0 / np.sqrt(alpha_t)) * (x_t - coeff * predicted_noise)

        if t > 0:
            sigma = np.sqrt(self.schedule.posterior_variance[t])
            z = np.random.randn(*x_t.shape)
            x_prev = mu + sigma * z
        else:
            x_prev = mu  # No noise at final step

        return x_prev

    def sample(self, shape: Tuple, num_steps: int = None) -> np.ndarray:
        """
        Generate new samples by running di full reverse process.
        """
        if num_steps is None:
            num_steps = self.schedule.T

        # Start from pure noise
        x = np.random.randn(*shape)

        print(f"\nGenerating sample of shape {shape}...")
        print(f"Starting from pure noise: mean={x.mean():.3f}, std={x.std():.3f}")

        # Reverse iterate
        step_size = max(1, self.schedule.T // num_steps)
        for t in range(self.schedule.T - 1, -1, -step_size):
            x = self.reverse_step(x, t)
            if t % 200 == 0:
                print(f"  t={t:>4}: mean={x.mean():.3f}, std={x.std():.3f}")

        print(f"Final sample: mean={x.mean():.3f}, std={x.std():.3f}")
        return x


# =============================================================================
# PART 5: CARIBBEAN DATA PATTERNS
# =============================================================================

def create_caribbean_data_patterns() -> Dict[str, np.ndarray]:
    """
    Create synthetic Caribbean-themed data patterns fi diffusion experiments.
    """
    np.random.seed(42)
    patterns = {}

    # Hurricane spiral pattern
    t = np.linspace(0, 4 * np.pi, 200)
    r = t / (4 * np.pi)
    hurricane_x = r * np.cos(t) + np.random.randn(200) * 0.05
    hurricane_y = r * np.sin(t) + np.random.randn(200) * 0.05
    patterns["hurricane_spiral"] = np.column_stack([hurricane_x, hurricane_y])

    # Island chain (Caribbean archipelago)
    island_centers = [(-0.8, 0.2), (-0.4, 0.5), (0.0, 0.6),
                      (0.3, 0.4), (0.5, 0.1), (0.7, -0.2)]
    island_points = []
    for cx, cy in island_centers:
        n = 30
        pts = np.column_stack([
            np.random.randn(n) * 0.08 + cx,
            np.random.randn(n) * 0.08 + cy
        ])
        island_points.append(pts)
    patterns["island_chain"] = np.vstack(island_points)

    # Trade route network (shipping lanes between islands)
    route_points = []
    for i in range(len(island_centers) - 1):
        start = np.array(island_centers[i])
        end = np.array(island_centers[i + 1])
        t_vals = np.linspace(0, 1, 20)
        for tv in t_vals:
            pt = start + tv * (end - start) + np.random.randn(2) * 0.03
            route_points.append(pt)
    patterns["trade_routes"] = np.array(route_points)

    return patterns


def demonstrate_diffusion_on_caribbean_data():
    """Full diffusion demonstration on Caribbean patterns."""
    print("=" * 60)
    print(" DIFFUSION ON CARIBBEAN DATA PATTERNS")
    print("=" * 60)

    patterns = create_caribbean_data_patterns()

    for name, data in patterns.items():
        print(f"\n--- Pattern: {name} ---")
        print(f"  Shape: {data.shape}")
        print(f"  Mean:  ({data[:, 0].mean():.3f}, {data[:, 1].mean():.3f})")
        print(f"  Std:   ({data[:, 0].std():.3f}, {data[:, 1].std():.3f})")

    # Run forward diffusion on hurricane pattern
    schedule = NoiseSchedule(num_timesteps=1000)
    forward = ForwardDiffusion(schedule)
    forward.demonstrate_forward_process(patterns["hurricane_spiral"])

    # Demonstrate reverse process (untrained - just shows structure)
    denoiser = SimpleDenoiser(data_dim=2, hidden_dim=32)
    reverse = ReverseDiffusion(schedule, denoiser)

    # Simulate training losses
    print("\n--- Simulated Training Progress ---")
    for epoch in range(5):
        losses = []
        for _ in range(50):
            idx = np.random.randint(len(patterns["hurricane_spiral"]))
            x_0 = patterns["hurricane_spiral"][idx]
            loss = denoiser.training_step(x_0, schedule)
            losses.append(loss)
        print(f"  Epoch {epoch + 1}: avg_loss = {np.mean(losses):.4f}")

    # Generate (untrained model, so results will be noisy)
    sample = reverse.sample(shape=(10, 2), num_steps=50)
    print(f"\nGenerated {sample.shape[0]} points (untrained model - just demonstrating structure)")


# =============================================================================
# PART 6: QUIZ
# =============================================================================

QUIZ_QUESTIONS = """
=============================================================================
 QUIZ: DIFFUSION MODELS (10 Questions)
=============================================================================

Q1: What is the forward process in DDPM?
    a) Generating images from noise
    b) Gradually adding Gaussian noise to data over T timesteps
    c) Training the neural network
    d) Compressing the data

Q2: What does the reverse process learn to do?
    a) Add more noise
    b) Predict and remove noise iteratively to generate new data
    c) Classify images
    d) Compress data

Q3: What is the key advantage of the forward process formula
    q(x_t | x_0) using alpha_bar_t?
    a) It is more accurate
    b) It allows jumping directly to any timestep without iterating
    c) It uses less memory
    d) It produces better images

Q4: What does the neural network in DDPM actually predict?
    a) The clean data x_0
    b) The noise epsilon that was added at timestep t
    c) The timestep t
    d) The beta schedule

Q5: Why is the variance schedule important in diffusion models?
    a) It controls the model architecture
    b) It determines how quickly noise is added and thus the difficulty
       of the reverse denoising task
    c) It sets the learning rate
    d) It determines the output resolution

Q6: What is the training objective (loss function) for DDPM?
    a) Cross-entropy between predicted and actual images
    b) Mean squared error between predicted noise and actual noise
    c) KL divergence between distributions
    d) Wasserstein distance

Q7: For Caribbean hurricane track prediction, how could diffusion
    models be applied?
    a) Only for generating images of hurricanes
    b) Generating probabilistic ensemble forecasts by sampling multiple
       possible future tracks from a learned distribution
    c) Classifying hurricane categories
    d) Compressing satellite imagery

Q8: How many timesteps T are typically used in DDPM?
    a) 10
    b) 100
    c) 1000
    d) 10000

Q9: What is the relationship between diffusion models and score matching?
    a) They are unrelated
    b) The noise prediction network implicitly learns the score function
       (gradient of log probability), connecting DDPM to score-based models
    c) Score matching is faster
    d) Diffusion models replaced score matching

Q10: What is classifier-free guidance in modern diffusion models?
    a) Training without any labels
    b) A technique that interpolates between conditional and unconditional
       generation to control sample quality vs diversity
    c) Removing the classifier from the pipeline
    d) Using a separate classifier network
"""


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" CARIBBEAN AI ACADEMY - GRADUATES MODULE")
    print(" Lesson 03: Diffusion Models (DDPM) from Scratch")
    print(" By Adrian Dunkley")
    print("=" * 60)

    # Show noise schedule
    schedule = NoiseSchedule()
    schedule.show_schedule()

    # Demonstrate on Caribbean data
    demonstrate_diffusion_on_caribbean_data()

    # Quiz
    print(QUIZ_QUESTIONS)

    print("\n" + "=" * 60)
    print(" From noise to creation - diffusion is pure magic!")
    print(" Imagine generating Caribbean art, fashion, architecture...")
    print(" Next: MLOps - deploying yuh models fi real-world use!")
    print("=" * 60)

# 🧩 Puzzle Piece 9/12: The animal is known for its color... not blue, not red, but ___
