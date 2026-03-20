"""
Caribbean AI Academy — Forms 4-5: Neural Networks from Scratch
===============================================================
Lesson 6: Build a Neural Network with ONLY numpy!
Designed by Adrian Dunkley (https://Adriandunkley.net) — FREE!

Wi building a brain fi di computer today — no PyTorch, no TensorFlow,
just pure numpy and understanding!
"""

import numpy as np

print("""
╔══════════════════════════════════════════════════════════╗
║  LESSON 6: NEURAL NETWORKS FROM SCRATCH                 ║
║  Build yuh own brain fi di computer! 🧠                 ║
╚══════════════════════════════════════════════════════════╝

A neural network is inspired by how yuh brain works:
- Neurons receive signals (inputs)
- They process the signals (weighted sum + activation)
- They fire or don't fire (output)

Today wi gonna build one from SCRATCH — no libraries except numpy!
""")

# ============================================================
# PART 1: The Building Blocks
# ============================================================

print("=" * 55)
print("PART 1: ACTIVATION FUNCTIONS")
print("=" * 55)

def sigmoid(x):
    """Squashes any number to between 0 and 1.
    Like a dimmer switch — smoothly goes from off (0) to on (1)."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """How fast sigmoid is changing — needed for learning."""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """If positive, pass through. If negative, output 0.
    Like a one-way valve — only lets positive signals through."""
    return np.maximum(0, x)

# Demo
print("\nSigmoid examples:")
for val in [-5, -1, 0, 1, 5]:
    print(f"  sigmoid({val:2d}) = {sigmoid(val):.4f}")

print("\nReLU examples:")
for val in [-5, -1, 0, 1, 5]:
    print(f"  relu({val:2d}) = {relu(val):.1f}")


# ============================================================
# PART 2: XOR Problem — The Classic Challenge
# ============================================================

print("\n" + "=" * 55)
print("PART 2: SOLVING XOR (The Problem That Changed AI)")
print("=" * 55)
print("""
XOR (exclusive or) — like a Caribbean weather rule:
  - Sunny + Dry    = 0 (no sunburn if yuh inside)
  - Sunny + Humid  = 1 (yuh getting sunburned!)
  - Cloudy + Dry   = 1 (surprise UV through clouds!)
  - Cloudy + Humid = 0 (safe — cloud cover + moisture)

A single neuron CANNOT learn XOR. That's why we need LAYERS!
""")

# XOR data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Neural network: 2 inputs -> 4 hidden -> 1 output
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5   # Input to hidden weights
b1 = np.zeros((1, 4))                # Hidden biases
W2 = np.random.randn(4, 1) * 0.5   # Hidden to output weights
b2 = np.zeros((1, 1))                # Output bias

learning_rate = 1.0
losses = []

print("Training neural network on XOR...")
for epoch in range(5000):
    # === FORWARD PASS ===
    # Like water flowing through pipes — input goes in, prediction comes out
    hidden_input = X_xor @ W1 + b1      # Weighted sum at hidden layer
    hidden_output = sigmoid(hidden_input) # Activation
    final_input = hidden_output @ W2 + b2 # Weighted sum at output
    prediction = sigmoid(final_input)      # Final prediction

    # === LOSS CALCULATION ===
    # How wrong are we? (Mean Squared Error)
    loss = np.mean((y_xor - prediction) ** 2)
    losses.append(loss)

    # === BACKWARD PASS (BACKPROPAGATION) ===
    # Like tracing back through di pipes to find where di leak is
    output_error = prediction - y_xor
    output_delta = output_error * sigmoid_derivative(final_input)

    hidden_error = output_delta @ W2.T
    hidden_delta = hidden_error * sigmoid_derivative(hidden_input)

    # === UPDATE WEIGHTS ===
    # Adjust di pipes to reduce di leak
    W2 -= learning_rate * hidden_output.T @ output_delta / len(X_xor)
    b2 -= learning_rate * np.mean(output_delta, axis=0, keepdims=True)
    W1 -= learning_rate * X_xor.T @ hidden_delta / len(X_xor)
    b1 -= learning_rate * np.mean(hidden_delta, axis=0, keepdims=True)

    if epoch % 1000 == 0:
        print(f"  Epoch {epoch:4d} | Loss: {loss:.6f}")

print(f"  Epoch {epoch:4d} | Loss: {loss:.6f}")

# Test
print("\nXOR Results:")
print(f"  Input [0,0] → {prediction[0][0]:.4f} (expected 0)")
print(f"  Input [0,1] → {prediction[1][0]:.4f} (expected 1)")
print(f"  Input [1,0] → {prediction[2][0]:.4f} (expected 1)")
print(f"  Input [1,1] → {prediction[3][0]:.4f} (expected 0)")
print("  Neural network learned XOR! A single neuron couldn't do this!")


# ============================================================
# PART 3: Caribbean Crop Disease Classification
# ============================================================

print("\n" + "=" * 55)
print("PART 3: CARIBBEAN CROP DISEASE CLASSIFIER")
print("=" * 55)
print("""
Caribbean farmers need help detecting crop diseases early!
We'll build a neural network to classify whether sugarcane,
cocoa, or banana crops are HEALTHY or DISEASED based on features.

Features:
- Leaf color score (0-1, 1 = healthy green)
- Moisture level (0-1)
- Spot count (normalized, more spots = possibly diseased)
- Growth rate (0-1, 1 = normal growth)

Countries where dis matters: Jamaica, Guyana, Belize,
Trinidad, Barbados, Grenada, St. Vincent, Dominica...
basically ALL of di Caribbean!
""")

np.random.seed(42)
n_samples = 200

# Generate synthetic crop data
# Healthy crops: high color, good moisture, few spots, good growth
healthy_features = np.column_stack([
    np.random.normal(0.8, 0.1, n_samples//2),  # Leaf color (green)
    np.random.normal(0.7, 0.15, n_samples//2),  # Moisture
    np.random.normal(0.2, 0.1, n_samples//2),  # Spot count (low)
    np.random.normal(0.8, 0.1, n_samples//2),  # Growth rate
])

# Diseased crops: low color, variable moisture, many spots, slow growth
diseased_features = np.column_stack([
    np.random.normal(0.4, 0.15, n_samples//2),  # Leaf color (yellow/brown)
    np.random.normal(0.5, 0.2, n_samples//2),   # Moisture (variable)
    np.random.normal(0.7, 0.15, n_samples//2),  # Spot count (high)
    np.random.normal(0.3, 0.15, n_samples//2),  # Growth rate (slow)
])

X = np.clip(np.vstack([healthy_features, diseased_features]), 0, 1)
y = np.array([[0]]*100 + [[1]]*100)  # 0=healthy, 1=diseased

# Shuffle
indices = np.random.permutation(200)
X = X[indices]
y = y[indices]

# Split
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Build neural network: 4 inputs -> 8 hidden -> 4 hidden -> 1 output
W1 = np.random.randn(4, 8) * 0.5
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 4) * 0.5
b2 = np.zeros((1, 4))
W3 = np.random.randn(4, 1) * 0.5
b3 = np.zeros((1, 1))

learning_rate = 0.5

print("\nTraining crop disease classifier...")
for epoch in range(3000):
    # Forward pass
    z1 = X_train @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    z3 = a2 @ W3 + b3
    prediction = sigmoid(z3)

    # Loss
    loss = np.mean((y_train - prediction) ** 2)

    # Backpropagation
    d3 = (prediction - y_train) * sigmoid_derivative(z3)
    d2 = (d3 @ W3.T) * sigmoid_derivative(z2)
    d1 = (d2 @ W2.T) * sigmoid_derivative(z1)

    # Update
    n = len(X_train)
    W3 -= learning_rate * a2.T @ d3 / n
    b3 -= learning_rate * np.mean(d3, axis=0, keepdims=True)
    W2 -= learning_rate * a1.T @ d2 / n
    b2 -= learning_rate * np.mean(d2, axis=0, keepdims=True)
    W1 -= learning_rate * X_train.T @ d1 / n
    b1 -= learning_rate * np.mean(d1, axis=0, keepdims=True)

    if epoch % 1000 == 0:
        print(f"  Epoch {epoch:4d} | Loss: {loss:.6f}")

# Test
z1_test = X_test @ W1 + b1
a1_test = sigmoid(z1_test)
z2_test = a1_test @ W2 + b2
a2_test = sigmoid(z2_test)
z3_test = a2_test @ W3 + b3
test_pred = sigmoid(z3_test)

test_accuracy = np.mean((test_pred > 0.5).astype(int) == y_test)
print(f"\nTest Accuracy: {test_accuracy:.1%}")

# Sample predictions
print("\nSample Predictions:")
crop_types = ["Sugarcane", "Cocoa", "Banana", "Nutmeg", "Coffee"]
for i in range(5):
    pred_label = "DISEASED" if test_pred[i] > 0.5 else "HEALTHY"
    actual_label = "DISEASED" if y_test[i] == 1 else "HEALTHY"
    crop = crop_types[i % len(crop_types)]
    conf = test_pred[i][0] if test_pred[i] > 0.5 else 1 - test_pred[i][0]
    print(f"  {crop:12s} → Predicted: {pred_label:8s} (conf: {conf:.1%}) | Actual: {actual_label}")

print("""
DIS IS POWERFUL! 💪

Imagine: a farmer in rural Jamaica or Guyana takes a photo of their crop.
An app extracts features (leaf color, spots, etc.) and feeds it through
a neural network like dis one. Instantly dem know if di crop sick!

Early detection = save di crop = save di farmer's livelihood.
Dat's how AI can help di Caribbean directly.
""")


# ============================================================
# QUIZ
# ============================================================

print("=" * 55)
print("INTERNALIZATION QUIZ")
print("=" * 55)
print("""
Q1: What is an activation function?
    a) A function that activates the computer
    b) A function that introduces non-linearity into the network
    c) A function that turns the computer on
    d) A function that counts neurons

Q2: What does sigmoid do to its input?
    a) Doubles it  b) Squares it
    c) Squashes it between 0 and 1  d) Makes it negative

Q3: Why couldn't a single neuron solve XOR?
    a) XOR is too simple  b) XOR is not linearly separable
    c) Neurons can't do math  d) XOR doesn't exist

Q4: What is backpropagation?
    a) Going backward in code
    b) Computing gradients to figure out how to adjust weights
    c) Backing up your data
    d) Reversing the input

Q5: In our crop disease example, what were the 4 input features?
    a) Name, size, price, color
    b) Leaf color, moisture, spot count, growth rate
    c) Weight, height, age, type
    d) Temperature, rain, sun, wind

Q6: How many layers did our crop disease network have?
    a) 1  b) 2  c) 3 (input, 2 hidden, output = 4 including input)  d) 10

Q7: What is the learning rate?
    a) How fast the model runs
    b) How big the weight adjustment steps are
    c) How many epochs to train
    d) The speed of the computer

Q8: How could this crop disease classifier help Caribbean farmers?
    (Open-ended — think about early detection, mobile apps, etc.)

Answers in quiz_answers.md
""")
