# Caribbean AI Academy - Sixth Form
# Quiz Answers for Lessons 1-6

**Author: Adrian Dunkley | Caribbean AI Academy**

---

## Lesson 01: Deep Learning Fundamentals (CNNs)

**Q1:** b) Convolutional Neural Network — image/spatial data

**Q2:** b) It slides across the image, multiplying and summing values to detect features

**Q3:** c) To downsample feature maps, reducing size while keeping important features

**Q4:** a) To introduce non-linearity so the network can learn complex patterns

**Q5:** b) Randomly deactivates neurons during training to prevent overfitting

**Q6:** b) 4x4
- 16x16 -> MaxPool(2,2) -> 8x8 -> MaxPool(2,2) -> 4x4

**Q7:** b) CNNs automatically learn relevant features from raw images without manual feature engineering

**Q8:** b) One complete pass through the entire training dataset

**Q9:** b) Normalizes layer inputs to stabilize and speed up training

**Q10:** b) Use transfer learning with a pre-trained model and fine-tune on reef images

---

## Lesson 02: Recurrent Neural Networks & LSTMs

**Q1:** b) It treats each input independently with no memory of sequence

**Q2:** b) Vanishing gradient — they forget long-term dependencies

**Q3:** b) Forget, Input, Output

**Q4:** b) Because dengue has seasonal patterns that need several months of context to capture

**Q5:** b) Neural networks train better with values in a small range (like 0-1); raw values cause instability

**Q6:** b) MSELoss (Mean Squared Error) — because we're predicting a continuous number (case count), not a category

**Q7:** b) Actual rainfall data, Aedes aegypti mosquito counts, previous outbreak severity, population density

**Q8:** b) Input tensor shape is (batch_size, seq_length, features) instead of (seq_length, batch_size, features)

**Q9:** b) Stacking layers lets the model learn hierarchical temporal patterns — short-term in layer 1, longer-term in layer 2

**Q10:** b) Share standardized data across islands, train a regional model that captures Caribbean-wide patterns while allowing local fine-tuning for each territory

---

## Lesson 03: Transfer Learning

**Q1:** b) Using knowledge learned from one task to improve performance on a different but related task

**Q2:** b) Caribbean datasets are often small due to smaller populations, so leveraging pre-trained models saves time and improves results

**Q3:** b) Economic sector classification (with 500 samples)

**Q4:** b) Taking a pre-trained model and retraining some layers on new data

**Q5:** b) Use a model pre-trained on ImageNet, freeze early layers, fine-tune later layers on the 200 coral images

**Q6:** b) Adapting a model trained in one domain (e.g., US health data) to work well in a different domain (e.g., Caribbean health data)

**Q7:** b) We used the source Random Forest's leaf node indices as additional features for the target model

**Q8:** a) The source and target domains are too different (negative transfer)

**Q9:** b) Base skills (fitness, coordination) transfer, but sport-specific skills still need to be learned — similar to how general features transfer but task-specific layers need retraining

**Q10:** Open-ended. Strong answers include:
- Medical imaging (train on global X-rays, fine-tune on Caribbean patient scans for tropical diseases)
- Crop disease detection (train on global crop datasets, fine-tune on Caribbean crops like ackee, dasheen, breadfruit)
- Language processing (adapt English NLP models for Patois/Creole)
- Marine species identification (global fish models adapted for Caribbean reef species)
- Hurricane damage assessment (global disaster imagery adapted for Caribbean building styles)

---

## Lesson 04: Reinforcement Learning

**Q1:** b) Agent, Environment, State, Action, Reward

**Q2:** b) A lookup table storing expected cumulative rewards for each state-action pair

**Q3:** b) With probability epsilon, take a random action (explore); otherwise take the best known action (exploit)

**Q4:** b) Early on, the agent needs to explore to discover good strategies; later, it should exploit what it learned

**Q5:** b) To encourage the agent to find the SHORTEST path to the port rather than wandering around forever

**Q6:** b) It determines how much the agent values future rewards vs. immediate rewards; gamma=0.95 means future rewards are almost as important as immediate ones

**Q7:** b) Like humans, it updates beliefs based on new experience — blending what yuh already know with what yuh just observed

**Q8:** b) The magnitude of rewards/penalties shapes the agent's priorities; a large negative reward strongly discourages entering hurricane zones, modelling real catastrophic risk

**Q9:** b) The state space would increase from 64 to 256, requiring more episodes to explore and learn — but the algorithm stays the same

**Q10:** Open-ended. Strong answers include:
- Hurricane evacuation routing: State=traffic/road conditions, Action=route choices, Reward=people safely evacuated
- Energy grid management: State=demand/supply/weather, Action=which power source to use, Reward=cost savings + green energy usage
- Fisheries management: State=fish population/season, Action=fishing quotas, Reward=sustainable catch maximised
- Traffic light control: State=vehicle counts at intersection, Action=light timing, Reward=reduced wait times

---

## Lesson 05: Generative Adversarial Networks (GANs)

**Q1:** b) Generator and Discriminator

**Q2:** b) To create fake data realistic enough to fool the Discriminator

**Q3:** b) To correctly distinguish real data from generated (fake) data

**Q4:** b) To produce output in the [-1, 1] range, matching our normalized data

**Q5:** b) When the Generator produces only a few types of output instead of diverse samples — it found one trick to fool the Discriminator and keeps using it

**Q6:** b) It can fill data gaps, protect privacy, enable policy simulations, and augment small datasets for AI training

**Q7:** b) A random seed that the Generator transforms into structured data — each noise vector produces a different economic profile

**Q8:** b) Both Generator and Discriminator losses are balanced, and D(fake) approaches 0.5 (Discriminator can't tell real from fake)

**Q9:** b) Generate additional synthetic patient records that match the statistical properties of the real data, expanding the dataset for AI training while preserving privacy

**Q10:** b) Potential for misuse (fake statistics), bias amplification if training data is biased, transparency about what is synthetic vs real, and ensuring generated data doesn't reinforce harmful stereotypes about Caribbean nations

---

## Lesson 06: AI Ethics and Bias

**Q1:** b) When an AI system produces unfair or discriminatory results due to biased data, design, or deployment

**Q2:** b) Training datasets are dominated by lighter-skinned faces, so the model has less experience with darker skin tones common in the Caribbean

**Q3:** b) The principle that Caribbean nations and peoples should have control over data generated within and about their communities

**Q4:** b) AI could replace some roles (check-in, booking) but Caribbean hospitality's human warmth, cultural authenticity, and spontaneity are irreplaceable competitive advantages

**Q5:** b) Foreign technology companies extracting data and value from Caribbean nations similar to how colonial powers extracted natural resources

**Q6:** b) Through speech recognition, automated transcription, AI-powered dictionaries, and NLP tools adapted for Patois, Kreyol, Papiamento, and other Caribbean languages

**Q7:** b) Cultural appropriation — Caribbean artistic expressions created without Caribbean involvement or compensation, potentially replacing authentic Caribbean artists

**Q8:** b) Caribbean nations bear disproportionate climate impacts; AI climate tools should serve Caribbean resilience rather than just benefiting insurance companies or wealthy nations

**Q9:** b) If Caribbean data and contexts are used to create AI value, the resulting benefits (economic, social, technological) should flow back to Caribbean communities

**Q10:** b) To build AI systems that serve Caribbean communities equitably, question bias in AI tools, bring Caribbean perspectives to global AI development, and ensure technology uplifts rather than exploits our people

---

*Caribbean AI Academy | Designed by Adrian Dunkley*
*Walk good and study hard! The Caribbean needs YOU in AI!*
