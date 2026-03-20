"""
Caribbean AI Academy — Deep Learning Examples
================================================
PyTorch examples with Caribbean context.
Designed by Adrian Dunkley (https://Adriandunkley.net) — FREE!
"""

import numpy as np

# ============================================================
# EXAMPLE 1: Caribbean Image Classification CNN
# ============================================================

def caribbean_cnn_example():
    """
    Build a CNN to classify Caribbean land use from satellite-like features.
    Categories: coastal, reef, urban, agricultural, forest
    Using synthetic data so it runs without real images.
    """
    print("=" * 60)
    print("EXAMPLE 1: Caribbean Land Use Classification (CNN Concept)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch")
        print("Showing architecture only...\n")
        print("""
class CaribbeanLandCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 5)  # 5 Caribbean land types
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32->16
        x = self.pool(self.relu(self.conv2(x)))  # 16->8
        x = self.pool(self.relu(self.conv3(x)))  # 8->4
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)
        """)
        return

    # Land use categories across Caribbean islands
    categories = ["Coastal", "Reef", "Urban", "Agricultural", "Forest"]
    islands = ["Jamaica", "Trinidad", "Barbados", "Bahamas", "Grenada",
               "St. Lucia", "Dominica", "Guyana", "Belize", "Haiti"]

    print(f"Classifying land types for: {', '.join(islands)}")
    print(f"Categories: {', '.join(categories)}")

    # Generate synthetic "image" data (32x32 RGB)
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 3, 32, 32).astype(np.float32)

    # Add distinguishing features per class
    y = np.random.randint(0, 5, n_samples)
    for i in range(n_samples):
        if y[i] == 0:  # Coastal - blue tones
            X[i, 2] += 2.0  # Blue channel high
        elif y[i] == 1:  # Reef - cyan/green
            X[i, 1] += 1.5
            X[i, 2] += 1.0
        elif y[i] == 2:  # Urban - gray/high contrast
            X[i] += 1.0
        elif y[i] == 3:  # Agricultural - green
            X[i, 1] += 2.0
        elif y[i] == 4:  # Forest - dark green
            X[i, 1] += 2.5
            X[i, 0] -= 0.5

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Split train/test
    train_size = 400
    train_ds = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    test_ds = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    # Define CNN
    class CaribbeanLandCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 5)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    model = CaribbeanLandCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    print("\nTraining CNN...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            _, predicted = torch.max(output, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    print(f"\nTest Accuracy: {100*correct/total:.1f}%")
    print("Categories: Coastal, Reef, Urban, Agricultural, Forest")
    print("Real satellite images would give much better results!\n")


# ============================================================
# EXAMPLE 2: Caribbean Time Series LSTM
# ============================================================

def caribbean_lstm_example():
    """
    LSTM for predicting Caribbean tourism arrivals.
    Time series forecasting using monthly data.
    """
    print("=" * 60)
    print("EXAMPLE 2: Caribbean Tourism Forecasting (LSTM)")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed. Showing concept only.")
        return

    # Generate synthetic monthly tourism data for Barbados
    np.random.seed(42)
    months = 120  # 10 years of monthly data
    t = np.arange(months)

    # Tourism has seasonality (high season Dec-Apr) + trend + noise
    trend = t * 50  # Growing tourism
    seasonality = 15000 * np.sin(2 * np.pi * t / 12 - np.pi/3)  # Peak in winter
    noise = np.random.randn(months) * 3000

    # Add hurricane season dip (Aug-Oct)
    hurricane_dip = np.zeros(months)
    for i in range(months):
        month_of_year = i % 12
        if month_of_year in [7, 8, 9]:  # Aug, Sep, Oct
            hurricane_dip[i] = -8000

    tourism = 80000 + trend + seasonality + hurricane_dip + noise
    tourism = np.maximum(tourism, 10000)  # Floor

    print(f"Barbados monthly tourism data: {months} months (10 years)")
    print(f"Range: {tourism.min():.0f} to {tourism.max():.0f} visitors/month")

    # Prepare sequences
    def create_sequences(data, seq_length=12):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    # Normalize
    mean_val = tourism.mean()
    std_val = tourism.std()
    normalized = (tourism - mean_val) / std_val

    seq_len = 12
    X, y = create_sequences(normalized, seq_len)

    split = int(0.8 * len(X))
    X_train = torch.FloatTensor(X[:split]).unsqueeze(-1)
    y_train = torch.FloatTensor(y[:split])
    X_test = torch.FloatTensor(X[split:]).unsqueeze(-1)
    y_test = torch.FloatTensor(y[split:])

    # Define LSTM
    class TourismLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :]).squeeze()

    model = TourismLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train
    print("\nTraining LSTM...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/50, Loss: {loss.item():.6f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)

        # Convert back to actual numbers
        pred_actual = predictions.numpy() * std_val + mean_val
        true_actual = y_test.numpy() * std_val + mean_val

    mae = np.mean(np.abs(pred_actual - true_actual))
    print(f"\nTest MAE: {mae:.0f} visitors/month")
    print(f"That's about {mae/mean_val*100:.1f}% of average monthly visitors")

    # Show some predictions
    print("\nSample Predictions vs Actual (last 5 months):")
    for i in range(-5, 0):
        print(f"  Predicted: {pred_actual[i]:,.0f}  |  Actual: {true_actual[i]:,.0f}")


# ============================================================
# EXAMPLE 3: Simple Text Generation RNN
# ============================================================

def caribbean_text_generation():
    """
    Character-level text generation trained on Caribbean text.
    Generates Caribbean-style text one character at a time.
    """
    print("=" * 60)
    print("EXAMPLE 3: Caribbean Text Generation (Character RNN)")
    print("=" * 60)

    # Caribbean text corpus (small for demo)
    corpus = """
    The Caribbean is a region of beauty and resilience. From Jamaica to Trinidad,
    from Barbados to Guyana, the people share a common spirit of creativity.
    The warm waters of the Caribbean Sea connect islands that are home to
    diverse cultures, languages, and traditions. Cricket matches unite nations,
    carnival celebrations paint the streets with color, and the rhythm of
    reggae, soca, calypso, and dancehall moves the soul. The Caribbean has
    given the world Bob Marley, Usain Bolt, Rihanna, Brian Lara, and countless
    other icons. Small island states face big challenges — hurricanes, climate
    change, economic vulnerability — but the Caribbean spirit is unbreakable.
    From the Blue Mountains of Jamaica to the Pitons of St. Lucia, from the
    rainforests of Dominica to the beaches of Turks and Caicos, this region
    is paradise with purpose. The future of the Caribbean is bright, and
    artificial intelligence will play a key role in building that future.
    """

    # Build character vocabulary
    chars = sorted(set(corpus))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    vocab_size = len(chars)

    print(f"Corpus: {len(corpus)} characters, {vocab_size} unique characters")
    print(f"This is a demo — real models train on much larger corpora!\n")

    # Simple bigram model (character-level) — no PyTorch needed
    # Count character transitions
    transitions = np.zeros((vocab_size, vocab_size))
    for i in range(len(corpus) - 1):
        current = char_to_idx[corpus[i]]
        next_char = char_to_idx[corpus[i + 1]]
        transitions[current][next_char] += 1

    # Normalize to probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = transitions / row_sums

    # Generate text
    def generate(seed_char, length=200, temperature=1.0):
        result = seed_char
        current = char_to_idx.get(seed_char, 0)
        for _ in range(length):
            p = probs[current].copy()
            if temperature != 1.0:
                p = np.power(p, 1.0 / temperature)
                if p.sum() > 0:
                    p /= p.sum()
            if p.sum() == 0:
                next_idx = np.random.randint(vocab_size)
            else:
                next_idx = np.random.choice(vocab_size, p=p)
            result += idx_to_char[next_idx]
            current = next_idx
        return result

    print("Generated Caribbean text (bigram model):")
    print("-" * 40)
    np.random.seed(42)
    for temp in [0.5, 1.0, 1.5]:
        text = generate("T", 150, temperature=temp)
        print(f"\n  Temperature {temp}:")
        print(f"  {text}")

    print("\n\nNote: A proper LSTM/Transformer would generate MUCH better text!")
    print("This bigram model just shows the basic concept.")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CARIBBEAN AI ACADEMY — DEEP LEARNING EXAMPLES")
    print("  Designed by Adrian Dunkley (https://Adriandunkley.net)")
    print("=" * 60 + "\n")

    caribbean_cnn_example()
    print("\n")
    caribbean_lstm_example()
    print("\n")
    caribbean_text_generation()

    print("\n" + "=" * 60)
    print("  Deep learning examples complete!")
    print("  Next: Try the LLMs module for transformer-based models")
    print("=" * 60)
