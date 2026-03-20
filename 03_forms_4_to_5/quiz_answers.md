# Forms 4-5 Quiz Answers

## Lesson 1: Supervised Learning

1. b) Training a model on labeled data (input-output pairs)
2. c) Features are the input variables the model uses to make predictions
3. b) Classification assigns categories, regression predicts continuous values
4. b) Testing on data the model hasn't seen during training
5. a) Decision trees split data based on feature thresholds
6. b) An ensemble of many decision trees that vote on the answer
7. c) Overfitting means the model memorized training data but fails on new data
8. d) All of the above (more trees, max depth limits, more training data)

## Lesson 2: Unsupervised Learning

1. b) Finding patterns in data without labels
2. a) Group similar data points together
3. b) K is the number of clusters to create
4. c) The Elbow Method — look for where adding more clusters stops improving much
5. b) It can find clusters of any shape, not just spherical
6. a) A dimensionality reduction technique that finds the directions of maximum variance
7. Caribbean economic clusters: Tourism-heavy (Bahamas, Antigua), Diverse (Trinidad, Jamaica), Agriculture (Guyana, Belize)
8. b) It can handle varying cluster sizes and shapes

## Lesson 3: Regression

1. b) A line of best fit through data points
2. c) A curve instead of a straight line
3. b) Mean Squared Error — the average of squared differences between predicted and actual
4. c) R-squared — the proportion of variance explained by the model
5. b) Using polynomial of too high a degree, fitting the noise not the pattern
6. Jamaica and Guyana's sugarcane yields show different relationships with rainfall
7. c) It can help farmers plan irrigation and predict harvest amounts
8. Regression helps predict crop yields, tourism revenue, hurricane damage costs — all continuous values

## Lesson 4: Classification

1. b) Assigning data points to predefined categories
2. c) Tempo, energy, danceability, beats per minute — audio features
3. b) The proportion of correct predictions out of total predictions
4. c) A table showing true positives, false positives, true negatives, false negatives
5. b) Random Forest — it's robust and handles multiple classes well
6. Dancehall vs reggaeton might be confused because they share similar tempo and rhythm patterns
7. b) Some genres are inherently similar (soca/calypso share roots)
8. Music streaming, radio playlists, cultural preservation, music recommendation

## Lesson 5: NLP Basics

1. b) Breaking text into words/pieces
2. c) Determining if text is positive/negative
3. b) Term Frequency-Inverse Document Frequency
4. c) Stop words (common words like 'the', 'and')
5. d) All of the above (context, sarcasm, negation all cause issues)
6. Check the output — islands with all 5-star reviews rank highest
7. Tourism review analysis, government document processing, healthcare communication in local dialects, legal document summarization
8. b) Multiple languages and Creole dialects make NLP challenging

## Lesson 6: Neural Networks

1. b) A function that introduces non-linearity into the network
2. c) Squashes it between 0 and 1
3. b) XOR is not linearly separable (can't draw a single straight line to separate the classes)
4. b) Computing gradients to figure out how to adjust weights
5. b) Leaf color, moisture, spot count, growth rate
6. c) 3 weight matrices = 4 layers counting input: input(4) → hidden(8) → hidden(4) → output(1)
7. b) How big the weight adjustment steps are
8. Farmers could use a phone app that photographs crops and classifies diseases instantly. Early detection saves crops and livelihoods across Jamaica, Guyana, Trinidad, Belize, etc.
