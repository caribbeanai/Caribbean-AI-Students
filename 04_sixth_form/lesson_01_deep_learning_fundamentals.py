"""
=============================================================================
Caribbean AI Academy - Sixth Form (Ages 16-18)
Lesson 01: Deep Learning Fundamentals
=============================================================================
Topic: Convolutional Neural Networks (CNNs) for Caribbean Satellite Image
        Classification

Big up yuhself for reaching Sixth Form AI! In this lesson, we gon learn
how deep learning works — specifically CNNs — and use dem to classify
Caribbean satellite images into 5 categories:
  - Coastal (beaches, shoreline)
  - Reef (coral reef areas)
  - Urban (towns and cities like Kingston, Port of Spain, Nassau)
  - Agricultural (sugar cane, banana, cocoa fields)
  - Forest (rainforest, mangrove, tropical bush)

Think of a CNN like how yuh eyes work when yuh scanning the Caribbean
landscape from a plane — yuh brain pick up patterns: blue water = coast,
green canopy = forest, grey buildings = urban. CNNs do the same thing
with numbers!

We use synthetic (fake but realistic) feature data here since we cyah
download real satellite images in a lesson script. But the architecture
is REAL — same ting NASA and UWI researchers use fi study we islands.

Requirements: pip install torch numpy
Author: Adrian Dunkley | Caribbean AI Academy
=============================================================================
"""

# 🧩 Puzzle Piece 6/12: The weapon is not a gun, not a bow, but something a knight would carry

import numpy as np
import random

# =====================================================================
# SECTION 1: Understanding Convolutions (The Core of CNNs)
# =====================================================================
# A convolution is like a sliding window that scans across an image.
# Imagine yuh have a magnifying glass (called a "kernel" or "filter")
# and yuh slide it across a photo of Montego Bay beach. At each
# position, the kernel multiplies pixel values and sums them up.
# This helps detect features like edges, textures, and shapes.

print("=" * 65)
print("  LESSON 01: Deep Learning Fundamentals - CNN Classification")
print("  Caribbean Satellite Image Classification")
print("=" * 65)

# Let's demonstrate a simple convolution operation manually first
print("\n--- Part 1: What is a Convolution? ---\n")

# A tiny 5x5 "image" (think of it as a patch of satellite imagery)
# Higher values = brighter pixels
sample_image = np.array([
    [0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=float)

# An edge-detection kernel (3x3 filter)
# This particular kernel detects vertical edges — like the edge
# of a building in Kingston or the coastline in Barbados
edge_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)

print("Sample 5x5 image patch (like a tiny satellite tile):")
print(sample_image)
print("\nEdge detection kernel (3x3 filter):")
print(edge_kernel)


def simple_convolution(image, kernel):
    """
    Perform a basic 2D convolution without padding.
    This is what happens inside every CNN layer!

    Think of it like dragging a fish pot across the sea floor —
    at each spot, yuh catch (multiply and sum) what's underneath.
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)

    return output


conv_result = simple_convolution(sample_image, edge_kernel)
print("\nConvolution result (feature map):")
print(conv_result)
print("See how it highlights the edges? That's feature extraction!")


# =====================================================================
# SECTION 2: Building a CNN with PyTorch
# =====================================================================
print("\n--- Part 2: Building a CNN for Satellite Classification ---\n")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. We'll show the architecture conceptually.")
    print("Install with: pip install torch")

# Our 5 Caribbean land cover classes
CLASSES = ['Coastal', 'Reef', 'Urban', 'Agricultural', 'Forest']

# Caribbean locations for context — every island represented!
LOCATIONS = {
    'Coastal': ['Negril Beach JA', 'Maracas Bay TT', 'Eagle Beach Aruba',
                'Seven Mile Beach Cayman', 'Crane Beach Barbados'],
    'Reef': ['Belize Barrier Reef', 'Tobago Cays SVG', 'Buck Island USVI',
             'Glover\'s Reef Belize', 'Bonaire Marine Park'],
    'Urban': ['Kingston JA', 'Port of Spain TT', 'Nassau Bahamas',
              'Bridgetown Barbados', 'Havana Cuba', 'Santo Domingo DR'],
    'Agricultural': ['Trelawny Sugar Cane JA', 'Caroni Rice TT',
                     'Banana Fields St Lucia', 'Cocoa Estates Grenada'],
    'Forest': ['Blue Mountains JA', 'El Yunque PR', 'Northern Range TT',
               'Cockpit Country JA', 'Dominica Rainforest']
}


def generate_synthetic_satellite_data(num_samples=500, image_size=16):
    """
    Generate synthetic satellite image data for Caribbean land classification.

    Real satellite images have multiple spectral bands (RGB + infrared etc.).
    We simulate 3-channel (RGB) images where each class has distinct
    colour/texture patterns — just like real Caribbean landscapes:
      - Coastal: lots of blue and sandy tones
      - Reef: turquoise and green underwater patterns
      - Urban: grey with sharp edges
      - Agricultural: bright green, regular patterns
      - Forest: deep green, varied textures
    """
    images = []
    labels = []

    for _ in range(num_samples):
        label = random.randint(0, 4)
        img = np.zeros((3, image_size, image_size))

        if label == 0:  # Coastal — blue water, tan sand
            img[0] = np.random.normal(0.7, 0.1, (image_size, image_size))
            img[1] = np.random.normal(0.6, 0.1, (image_size, image_size))
            img[2] = np.random.normal(0.9, 0.1, (image_size, image_size))
        elif label == 1:  # Reef — turquoise tones
            img[0] = np.random.normal(0.3, 0.15, (image_size, image_size))
            img[1] = np.random.normal(0.7, 0.12, (image_size, image_size))
            img[2] = np.random.normal(0.8, 0.1, (image_size, image_size))
        elif label == 2:  # Urban — grey, high contrast
            base = np.random.normal(0.5, 0.2, (image_size, image_size))
            img[0] = base + np.random.normal(0, 0.05, (image_size, image_size))
            img[1] = base + np.random.normal(0, 0.05, (image_size, image_size))
            img[2] = base + np.random.normal(0, 0.05, (image_size, image_size))
        elif label == 3:  # Agricultural — bright green, regular
            img[0] = np.random.normal(0.3, 0.08, (image_size, image_size))
            img[1] = np.random.normal(0.8, 0.08, (image_size, image_size))
            img[2] = np.random.normal(0.2, 0.08, (image_size, image_size))
        else:  # Forest — deep green, varied
            img[0] = np.random.normal(0.15, 0.1, (image_size, image_size))
            img[1] = np.random.normal(0.6, 0.15, (image_size, image_size))
            img[2] = np.random.normal(0.1, 0.1, (image_size, image_size))

        img = np.clip(img, 0, 1)
        images.append(img)
        labels.append(label)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


# Generate our dataset
print("Generating synthetic Caribbean satellite data...")
X_data, y_data = generate_synthetic_satellite_data(num_samples=600)
print(f"Dataset shape: {X_data.shape} (samples, channels, height, width)")
print(f"Labels shape: {y_data.shape}")
print(f"Classes: {CLASSES}")

for i, cls in enumerate(CLASSES):
    count = np.sum(y_data == i)
    print(f"  {cls}: {count} samples (e.g., {LOCATIONS[cls][0]})")


if TORCH_AVAILABLE:
    # =================================================================
    # SECTION 3: Define the CNN Architecture
    # =================================================================
    print("\n--- Part 3: CNN Architecture ---\n")

    class CaribbeanSatelliteCNN(nn.Module):
        """
        CNN for classifying Caribbean satellite images.

        Architecture (like building a house, layer by layer):
        1. Conv Layer 1: Detect basic features (edges, colours)
        2. Conv Layer 2: Combine into patterns (reef vs urban grid)
        3. Fully Connected Layers: Final classification decision
        """

        def __init__(self, num_classes=5):
            super(CaribbeanSatelliteCNN, self).__init__()

            # First convolutional block
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )

            # Second convolutional block — deeper features
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )

            # After two pooling layers: 16x16 -> 8x8 -> 4x4
            self.fc1 = nn.Linear(32 * 4 * 4, 64)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            """Forward pass — data flows through the network like water
            flowing through a Caribbean river, refined at each stage."""
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = CaribbeanSatelliteCNN(num_classes=5)
    print("Model Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params:,}")

    # =================================================================
    # SECTION 4: Training the CNN
    # =================================================================
    print("\n--- Part 4: Training the CNN ---\n")

    split = 480
    X_train = torch.tensor(X_data[:split])
    y_train = torch.tensor(y_data[:split])
    X_test = torch.tensor(X_data[split:])
    y_test = torch.tensor(y_data[split:])

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop — like a West Indies cricket team practicing,
    # each epoch the model gets better!
    num_epochs = 15
    print(f"Training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        if (epoch + 1) % 3 == 0 or epoch == 0:
            acc = 100 * correct / total
            print(f"  Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {running_loss/len(train_loader):.4f}  "
                  f"Train Accuracy: {acc:.1f}%")

    # =================================================================
    # SECTION 5: Evaluation
    # =================================================================
    print("\n--- Part 5: Evaluating the Model ---\n")

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        test_acc = 100 * (predicted == y_test).sum().item() / y_test.size(0)

    print(f"Test Accuracy: {test_acc:.1f}%")
    print("\nPer-class results:")
    for i, cls in enumerate(CLASSES):
        mask = y_test == i
        if mask.sum() > 0:
            cls_acc = 100 * (predicted[mask] == y_test[mask]).sum().item() / mask.sum().item()
            print(f"  {cls}: {cls_acc:.1f}% accuracy")

    print("\nSample Predictions:")
    for idx in range(min(8, len(X_test))):
        true_label = CLASSES[y_test[idx].item()]
        pred_label = CLASSES[predicted[idx].item()]
        status = "CORRECT" if true_label == pred_label else "WRONG"
        print(f"  Sample {idx+1}: True={true_label}, "
              f"Predicted={pred_label} [{status}]")

else:
    print("\n[PyTorch not available — showing conceptual architecture]")
    print("""
    CNN Architecture for Caribbean Satellite Classification:
    --------------------------------------------------------
    Input: 3 x 16 x 16 image (RGB satellite tile)
        |
    Conv2d(3->16, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
        |   -> Detects edges, basic colour patterns
    Conv2d(16->32, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
        |   -> Detects complex patterns (reef textures, urban grids)
    Flatten -> Linear(512, 64) -> ReLU -> Dropout(0.3)
        |
    Linear(64, 5) -> Output (5 classes)
    """)


# =====================================================================
# SECTION 6: Caribbean Context & Real-World Applications
# =====================================================================
print("\n--- Part 6: Real-World Caribbean Applications ---\n")
print("""
CNNs like this are used across the Caribbean for:

1. CORAL REEF MONITORING (Belize, Bonaire, Tobago)
   - Satellite + drone images classified to track reef health
   - Critical for tourism and marine biodiversity

2. HURRICANE DAMAGE ASSESSMENT (All Caribbean nations)
   - After storms like Maria, Irma, Ivan — CNNs analyze satellite
     images to map destruction and guide relief efforts

3. DEFORESTATION TRACKING (Guyana, Suriname, Dominica)
   - Monitor illegal logging in the Amazon-Caribbean corridor

4. URBAN PLANNING (Kingston, Port of Spain, Santo Domingo)
   - Classify land use to plan infrastructure development

5. AGRICULTURAL MONITORING (Jamaica, Trinidad, Grenada)
   - Monitor crop health: sugar cane, bananas, cocoa, nutmeg
   - Like how a fast bowler reads the pitch, the CNN reads the land!

Real talk: Caribbean researchers at UWI, UTT, and CARICOM agencies
are actively using these techniques. This is YOUR future career path!
""")


# =====================================================================
# QUIZ: Test Yuh Knowledge!
# =====================================================================
print("=" * 65)
print("  QUIZ: Deep Learning Fundamentals")
print("=" * 65)
print("""
Answer these questions (write yuh answers in a notebook):

Q1: What does CNN stand for, and what type of data is it best for?
    a) Central Neural Network — tabular data
    b) Convolutional Neural Network — image/spatial data
    c) Connected Node Network — text data
    d) Caribbean Neural Network — island data

Q2: In a convolution, what does the "kernel" (filter) do?
    a) It deletes pixels from the image
    b) It slides across the image, multiplying and summing values
       to detect features
    c) It makes the image bigger
    d) It converts the image to black and white

Q3: What is the purpose of MaxPooling in a CNN?
    a) To make the image more colourful
    b) To add more pixels
    c) To downsample feature maps, reducing size while keeping
       important features
    d) To pool data from multiple Caribbean islands

Q4: Why do we use ReLU activation functions?
    a) To introduce non-linearity so the network can learn
       complex patterns
    b) To make all values negative
    c) To slow down training
    d) To reduce the number of parameters

Q5: What does the Dropout layer do?
    a) Drops images from the dataset
    b) Randomly deactivates neurons during training to prevent
       overfitting
    c) Removes the last layer of the network
    d) Drops the learning rate

Q6: If we have a 16x16 input and apply MaxPool2d(2,2) twice,
    what is the resulting spatial size?
    a) 8x8
    b) 4x4
    c) 2x2
    d) 16x16

Q7: Why would Caribbean researchers prefer CNNs for satellite
    image analysis?
    a) CNNs are cheaper to run
    b) CNNs automatically learn relevant features from raw images
       without manual feature engineering
    c) CNNs only work with Caribbean data
    d) Traditional methods don't exist

Q8: What is an "epoch" in neural network training?
    a) A unit of time measurement
    b) One complete pass through the entire training dataset
    c) The final accuracy score
    d) A type of activation function

Q9: What does Batch Normalization do?
    a) Batches data into island groups
    b) Normalizes layer inputs to stabilize and speed up training
    c) Removes batches with errors
    d) Makes all batches the same size

Q10: A marine biologist wants to classify coral reef health from
     underwater drone photos. What CNN approach would you suggest?
     a) Remove all convolutional layers
     b) Use transfer learning with a pre-trained model, fine-tune
        on reef images
     c) Use smaller images only
     d) Train on text data instead

(Answers in quiz_answers.md)
""")

print("=" * 65)
print("  Lesson 01 Complete! Yuh learn bout CNNs today — big tings!")
print("  Next up: Recurrent Neural Networks for Time Series")
print("=" * 65)
