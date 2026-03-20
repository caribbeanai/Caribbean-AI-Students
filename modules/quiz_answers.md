# Module Quiz Answers — Caribbean AI Academy

## Supervised Learning

1. c) Transformers
2. b) Learning from labeled input-output pairs
3. b) Classification assigns categories; regression predicts continuous values
4. b) Testing on unseen data to check if the model generalizes
5. c) Features the model learns are too specific to training data
6. b) Regularization, more data, simpler model, cross-validation
7. Open-ended: Dengue prediction, crop yield forecasting, student grade prediction, fish catch prediction, tourism demand forecasting
8. c) Mean Squared Error (MSE) for regression, Accuracy/F1 for classification
9. b) More trees = more robust predictions through voting
10. Open-ended: e.g., Predict hurricane damage severity based on storm features and island infrastructure

## Unsupervised Learning

1. b) Finding patterns and structure in unlabeled data
2. a) K-Means groups data into K clusters based on distance to centroids
3. c) The Elbow Method — plot inertia vs K, look for the "elbow" bend
4. b) DBSCAN can find clusters of arbitrary shape and detect outliers
5. a) PCA finds directions of maximum variance to reduce dimensionality
6. b) Anomaly detection in Caribbean financial transactions
7. Open-ended: Cluster Caribbean countries by economic indicators (tourism-heavy, diverse, agricultural)
8. c) It assigns probability of belonging to each cluster (soft clustering)
9. b) Finding groups of similar items bought together at Caribbean markets
10. Open-ended: e.g., Topic modeling on Caribbean news to find trending issues

## Reinforcement Learning

1. b) Learning through trial and error by maximizing rewards
2. c) Agent, Environment, State, Action, Reward
3. b) A table that maps (state, action) pairs to expected future rewards
4. b) Exploration tries new actions; exploitation uses known good actions
5. c) Epsilon-greedy — explore with probability epsilon, exploit otherwise
6. b) The discount factor — how much the agent values future vs immediate rewards
7. Open-ended: Navigating Caribbean shipping routes, optimizing bus schedules, managing hotel room pricing
8. b) A learned mapping from states to actions (replaces Q-table for continuous spaces)
9. c) Uses neural networks to approximate the Q-function for large state spaces
10. Open-ended: e.g., Optimize inter-island cargo shipping routes considering weather, fuel costs, and demand

## LLMs (Large Language Models)

1. c) Transformers
2. b) Focuses on relevant parts of input simultaneously
3. c) 6+ (English, Spanish, French, Dutch, Portuguese, and many Creoles)
4. b) Breaking text into smaller pieces the model can process
5. c) Underrepresented in training data, so tokens are fragmented
6. b) Retrieval Augmented Generation
7. b) Creativity/randomness of output
8. b) Helps model understand Caribbean context deeply
9. b) Western-centric training data underrepresents Caribbean perspectives
10. Open-ended: Digitizing oral traditions (Anansi stories), preserving Creole languages, creating searchable archives of Caribbean history and folk tales

## Diffusion Models

1. b) Forward (add noise) and reverse (remove noise)
2. b) Gradually adds noise until it becomes random
3. b) The noise that was added
4. c) DDPM (Denoising Diffusion Probabilistic Models)
5. b) Generate synthetic training images of crop diseases
6. b) Works in compressed latent space for efficiency
7. b) Darkness/randomness before the image forms
8. c) 100-1000 steps typically
9. b) Mean squared error between predicted and actual noise
10. Open-ended: Generate Carnival costume designs, create Caribbean tourism promotional images, augment small medical datasets for island hospitals

## Deep Learning

1. b) Many neural network layers (depth of the network)
2. c) CNN (Convolutional Neural Network)
3. b) Creates fake data to fool the Discriminator
4. b) Practicing against only one bowler (overfitting to specific patterns)
5. b) Step size during weight updates
6. Open-ended: Satellite image classification for coastline monitoring, crop disease detection from photos, coral reef health assessment
7. b) Computing gradients to update weights (learning from errors)
8. b) They remember long-term dependencies better (via cell state and gates)
9. b) Limited connectivity and power reliability on small islands
10. b) Detect crop diseases from images (computer vision for agriculture)
