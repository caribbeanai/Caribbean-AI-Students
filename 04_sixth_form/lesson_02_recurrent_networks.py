"""
=============================================================================
Caribbean AI Academy - Sixth Form (Ages 16-18)
Lesson 02: Recurrent Neural Networks (RNNs) & LSTMs
=============================================================================
Topic: LSTM for Dengue Fever Prediction in Trinidad & Tobago

Wah gwaan, future data scientists! Today we tackling a REAL Caribbean
problem — predicting dengue fever outbreaks using time series data.

Dengue is a massive public health issue across the Caribbean:
- Trinidad & Tobago, Jamaica, Barbados, Guyana, Suriname, Haiti —
  all affected every rainy season
- Prediction helps hospitals prepare, governments spray, and
  communities protect themselves

RNNs and LSTMs are special neural networks that understand SEQUENCES —
data where ORDER matters. Like how a cricket commentator tracks the
flow of an innings (1st over, 2nd over...), LSTMs track patterns
over time (January cases, February cases...).

Requirements: pip install torch numpy
Author: Adrian Dunkley | Caribbean AI Academy
=============================================================================
"""

import numpy as np
import random

print("=" * 65)
print("  LESSON 02: Recurrent Neural Networks & LSTMs")
print("  Dengue Fever Prediction in Trinidad & Tobago")
print("=" * 65)


# =====================================================================
# SECTION 1: Why Regular Neural Networks Can't Handle Sequences
# =====================================================================
print("\n--- Part 1: The Sequence Problem ---\n")
print("""
Imagine yuh tracking dengue cases in Port of Spain month by month:
  Jan: 50, Feb: 65, Mar: 120, Apr: 200, May: 350, Jun: ?

A regular neural network treats each input independently — it cyah
see that cases are RISING. But an RNN remembers previous inputs!

Think of it like this:
- Regular NN = A cricketer who forgets every ball after it's bowled
- RNN = A cricketer who remembers the whole innings and adjusts

The problem with basic RNNs? They have "short memory" — they forget
things from long ago (the "vanishing gradient" problem).

Solution: LSTM (Long Short-Term Memory) networks!
LSTMs have special "gates" that control what to remember and forget,
like how yuh brain remembers important tings (exam dates) but forgets
unimportant ones (what yuh had for lunch last Tuesday).
""")


# =====================================================================
# SECTION 2: Generate Synthetic Dengue Data
# =====================================================================
print("--- Part 2: Generating Synthetic Dengue Data ---\n")


def generate_dengue_data(num_years=10):
    """
    Generate synthetic monthly dengue case data for Trinidad & Tobago.

    Real patterns we model:
    - Seasonal: More cases in rainy season (Jun-Nov), fewer in dry season
    - Year-over-year trends: Climate change increasing cases
    - Random outbreaks: Some years worse than others
    - Related features: rainfall, temperature, humidity

    This mimics real data from CARPHA (Caribbean Public Health Agency)
    and TT Ministry of Health reports.
    """
    months = num_years * 12
    data = []

    for m in range(months):
        year = m // 12
        month_of_year = m % 12

        # Seasonal pattern — rainy season peaks (Jun=5 to Nov=10)
        # Like how West Indies cricket has home and away seasons
        seasonal = 80 * np.sin(2 * np.pi * (month_of_year - 3) / 12)
        seasonal = max(seasonal, 0)

        # Upward trend due to climate change and urbanisation
        trend = year * 8

        # Rainfall (mm) — correlates with dengue (mosquito breeding)
        rainfall = 150 + 100 * np.sin(2 * np.pi * (month_of_year - 4) / 12)
        rainfall += np.random.normal(0, 30)
        rainfall = max(rainfall, 20)

        # Temperature (Celsius) — Trinidad stays warm year-round
        temperature = 28 + 3 * np.sin(2 * np.pi * (month_of_year - 2) / 12)
        temperature += np.random.normal(0, 1)

        # Humidity (%)
        humidity = 70 + 15 * np.sin(2 * np.pi * (month_of_year - 4) / 12)
        humidity += np.random.normal(0, 5)
        humidity = np.clip(humidity, 40, 98)

        # Dengue cases — combination of all factors
        base_cases = 100 + seasonal + trend
        # Rainfall effect (more rain = more mosquitoes = more dengue)
        rain_effect = 0.3 * max(rainfall - 150, 0)
        # Random outbreak factor
        outbreak = 0
        if random.random() < 0.08:  # 8% chance of outbreak month
            outbreak = random.uniform(50, 200)

        cases = base_cases + rain_effect + outbreak
        cases += np.random.normal(0, 20)
        cases = max(int(cases), 5)

        data.append([rainfall, temperature, humidity, cases])

    return np.array(data, dtype=np.float32)


# Generate the data
raw_data = generate_dengue_data(num_years=10)
print(f"Generated {len(raw_data)} months of synthetic dengue data")
print(f"Features: Rainfall (mm), Temperature (C), Humidity (%), Cases")
print(f"\nSample data (first 6 months):")
print(f"{'Month':<8} {'Rain(mm)':<10} {'Temp(C)':<9} {'Humid(%)':<10} {'Cases':<8}")
print("-" * 45)
for i in range(6):
    print(f"{i+1:<8} {raw_data[i][0]:<10.1f} {raw_data[i][1]:<9.1f} "
          f"{raw_data[i][2]:<10.1f} {int(raw_data[i][3]):<8}")


# =====================================================================
# SECTION 3: Data Preprocessing for Time Series
# =====================================================================
print("\n--- Part 3: Preparing Data for LSTM ---\n")


def normalize_data(data):
    """Normalize to [0, 1] range — LSTMs work better with scaled data."""
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # avoid division by zero
    return (data - mins) / ranges, mins, maxs


def create_sequences(data, seq_length=6):
    """
    Create input sequences and targets for the LSTM.

    We use the past 'seq_length' months to predict next month's cases.
    Like a cricket analyst using the last 6 overs to predict the next!

    Example with seq_length=3:
      Input: [Jan, Feb, Mar] -> Target: Apr cases
      Input: [Feb, Mar, Apr] -> Target: May cases
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 3])  # Column 3 = dengue cases
    return np.array(X), np.array(y)


# Normalize the data
norm_data, data_mins, data_maxs = normalize_data(raw_data)
print(f"Data normalized to [0, 1] range")

# Create sequences — use 6 months of history to predict next month
SEQ_LENGTH = 6
X_seq, y_seq = create_sequences(norm_data, SEQ_LENGTH)
print(f"Sequence length: {SEQ_LENGTH} months (use past 6 months to predict)")
print(f"Total sequences: {len(X_seq)}")
print(f"Input shape: {X_seq.shape} (samples, timesteps, features)")
print(f"Target shape: {y_seq.shape}")


# =====================================================================
# SECTION 4: Building the LSTM Model
# =====================================================================
print("\n--- Part 4: LSTM Architecture ---\n")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Showing conceptual architecture.")

if TORCH_AVAILABLE:
    class DengueLSTM(nn.Module):
        """
        LSTM model for predicting dengue cases in Trinidad & Tobago.

        How LSTM works (simplified):
        - FORGET GATE: "Should I forget old information?"
          Like deciding whether last year's outbreak pattern still relevant
        - INPUT GATE: "What new information should I store?"
          Like noting that rainfall just increased dramatically
        - OUTPUT GATE: "What should I output right now?"
          Like the prediction: "Expect 250 cases next month"

        The "cell state" is like a conveyor belt running through time,
        carrying important information forward — like how Caribbean
        elders pass down knowledge through generations.
        """

        def __init__(self, input_size=4, hidden_size=32, num_layers=2,
                     output_size=1):
            super(DengueLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM layers — the heart of our model
            # input_size=4 because we have 4 features per month
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,      # Input shape: (batch, seq, features)
                dropout=0.2            # Regularization between LSTM layers
            )

            # Fully connected output layer
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            """
            Forward pass through the LSTM.
            x shape: (batch_size, seq_length, num_features)
            """
            # LSTM returns: output (all timesteps), (hidden_state, cell_state)
            lstm_out, (h_n, c_n) = self.lstm(x)

            # We only need the last timestep's output for prediction
            # Like only caring about the final score, not ball-by-ball
            last_output = lstm_out[:, -1, :]

            # Pass through fully connected layer
            prediction = self.fc(last_output)
            return prediction

    # Create model
    model = DengueLSTM(input_size=4, hidden_size=32, num_layers=2)
    print("LSTM Model Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # =================================================================
    # SECTION 5: Training
    # =================================================================
    print("\n--- Part 5: Training the LSTM ---\n")

    # Split data: 80% train, 20% test
    split_idx = int(len(X_seq) * 0.8)
    X_train = torch.tensor(X_seq[:split_idx])
    y_train = torch.tensor(y_seq[:split_idx]).unsqueeze(1)
    X_test = torch.tensor(X_seq[split_idx:])
    y_test = torch.tensor(y_seq[split_idx:]).unsqueeze(1)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    num_epochs = 50
    print(f"\nTraining for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_loss = criterion(test_pred, y_test)
            print(f"  Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {loss.item():.6f}  "
                  f"Test Loss: {test_loss.item():.6f}")

    # =================================================================
    # SECTION 6: Evaluation & Predictions
    # =================================================================
    print("\n--- Part 6: Predictions ---\n")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy().flatten()
        actual = y_test.numpy().flatten()

    # Convert back to actual case numbers
    case_range = data_maxs[3] - data_mins[3]
    pred_cases = predictions * case_range + data_mins[3]
    actual_cases = actual * case_range + data_mins[3]

    print(f"{'Month':<8} {'Actual':<12} {'Predicted':<12} {'Error':<10}")
    print("-" * 42)
    for i in range(min(12, len(pred_cases))):
        error = abs(pred_cases[i] - actual_cases[i])
        print(f"{i+1:<8} {actual_cases[i]:<12.0f} "
              f"{pred_cases[i]:<12.0f} {error:<10.0f}")

    mae = np.mean(np.abs(pred_cases - actual_cases))
    print(f"\nMean Absolute Error: {mae:.1f} cases")
    print(f"(On average, predictions off by ~{mae:.0f} cases per month)")

else:
    print("""
    LSTM Architecture (Conceptual):
    --------------------------------
    Input: (batch, 6 timesteps, 4 features)
        |
    LSTM Layer 1 (hidden_size=32, dropout=0.2)
        |   -> Learns short-term patterns (monthly trends)
    LSTM Layer 2 (hidden_size=32)
        |   -> Learns longer-term patterns (seasonal cycles)
    Linear(32, 1)
        |
    Output: Predicted dengue cases for next month
    """)


# =====================================================================
# SECTION 7: Caribbean Public Health Applications
# =====================================================================
print("\n--- Part 7: Caribbean Public Health Applications ---\n")
print("""
This kind of LSTM prediction model is used across the Caribbean:

1. DENGUE EARLY WARNING (Trinidad, Jamaica, Barbados, Guyana)
   - CARPHA uses similar models to predict outbreaks 1-3 months ahead
   - Hospitals can stock up on supplies, blood products

2. HURRICANE TRAJECTORY (All Caribbean)
   - LSTM models help predict storm paths using historical data
   - Like predicting where Usain Bolt gon finish — but with weather!

3. TOURISM DEMAND FORECASTING (Bahamas, Barbados, Jamaica)
   - Predict hotel occupancy, flight demand month by month
   - Critical for small island economies

4. CROP YIELD PREDICTION (Guyana rice, Jamaica coffee, Grenada nutmeg)
   - Time series of weather + soil data predicts harvest yields
   - Helps farmers plan, like a cricket captain setting the field

5. SEA LEVEL RISE TRACKING (All low-lying Caribbean nations)
   - LSTMs analyze decades of tide gauge data
   - Predict future sea levels for Barbuda, Bahamas, Belize atolls

6. COVID/DISEASE TRACKING (Region-wide)
   - Same architecture adapted for COVID-19 case prediction
   - Used by CARPHA, UWI School of Public Health
""")


# =====================================================================
# QUIZ
# =====================================================================
print("=" * 65)
print("  QUIZ: Recurrent Neural Networks & LSTMs")
print("=" * 65)
print("""
Q1: Why can't a regular (feedforward) neural network handle time
    series data well?
    a) It's too slow
    b) It treats each input independently with no memory of sequence
    c) It only works with images
    d) It requires too much data

Q2: What problem do basic RNNs suffer from that LSTMs solve?
    a) They are too fast
    b) Vanishing gradient — they forget long-term dependencies
    c) They can only process one input
    d) They require GPUs

Q3: What are the three "gates" in an LSTM cell?
    a) Open, Close, Lock
    b) Forget, Input, Output
    c) Start, Middle, End
    d) Read, Write, Delete

Q4: In our dengue model, why do we use 6 months of history
    (seq_length=6) to predict the next month?
    a) Because 6 is a lucky number
    b) Because dengue has seasonal patterns that need several months
       of context to capture
    c) Because the data only has 6 months
    d) Because LSTMs can only handle 6 inputs

Q5: Why do we normalize the data before feeding it to the LSTM?
    a) To make it look pretty
    b) Neural networks train better with values in a small range
       (like 0-1); raw values (e.g., 200mm rain) cause instability
    c) To remove outliers
    d) Because PyTorch requires it

Q6: What loss function did we use, and why?
    a) CrossEntropyLoss — for classification
    b) MSELoss (Mean Squared Error) — because we're predicting a
       continuous number (case count), not a category
    c) Binary loss — because it's yes or no
    d) Hinge loss — for margin optimization

Q7: In the context of dengue prediction for Trinidad, what real-world
    features would improve the model?
    a) Cricket scores
    b) Actual rainfall data, Aedes aegypti mosquito counts, previous
       outbreak severity, population density
    c) Stock market data
    d) Social media likes

Q8: What does "batch_first=True" mean in our LSTM?
    a) The first batch is most important
    b) Input tensor shape is (batch_size, seq_length, features)
       instead of (seq_length, batch_size, features)
    c) Only the first batch is trained
    d) Batches are processed first-come-first-served

Q9: Why did we use 2 LSTM layers (num_layers=2)?
    a) One is not enough to pass the exam
    b) Stacking layers lets the model learn hierarchical temporal
       patterns — short-term in layer 1, longer-term in layer 2
    c) PyTorch requires at least 2
    d) To double the speed

Q10: How could CARICOM nations collaborate on a regional dengue
     prediction system using LSTMs?
     a) They can't — each country is different
     b) Share standardized data across islands, train a regional model
        that captures Caribbean-wide patterns while allowing local
        fine-tuning for each territory
     c) Just use Jamaica's data for everyone
     d) Only use temperature data

(Answers in quiz_answers.md)
""")

print("=" * 65)
print("  Lesson 02 Complete! Yuh now understand RNNs and LSTMs!")
print("  Next up: Transfer Learning")
print("=" * 65)
