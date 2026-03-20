# =============================================================================
# CARIBBEAN AI ACADEMY - GRADUATES MODULE (05)
# Quiz Answers - Lessons 1 through 7
# By Adrian Dunkley
# =============================================================================

> "Knowledge without verification is just vibes. Check yuh answers!"

---

## Lesson 01: Transformers & Self-Attention

| Question | Answer | Explanation |
|----------|--------|-------------|
| Q1 | **b) Query, Key, Value** | Q asks "what am I looking for?", K says "what do I contain?", V says "what info do I give?" |
| Q2 | **b) Prevent dot products from growing too large** | Large dot products push softmax into regions with near-zero gradients, making training difficult |
| Q3 | **b) Give the model information about token positions** | Attention is permutation-invariant; without PE, "Jamaica loves cricket" = "cricket loves Jamaica" |
| Q4 | **c) 64** | d_k = d_model / num_heads = 512 / 8 = 64 per head |
| Q5 | **b) Attending to future tokens** | Prevents information leakage during autoregressive generation |
| Q6 | **b) Allow gradients to flow directly** | x + f(x) means gradient of 1 always flows through, preventing vanishing gradients |
| Q7 | **c) Feature/embedding dimension** | Unlike BatchNorm (batch dim), LayerNorm normalizes across features for each token |
| Q8 | **b) Every token computes attention with every other token** | The QK^T matrix is (n x n), giving O(n^2) complexity |
| Q9 | **c) 512** | The original Transformer paper used d_model=512 with 8 heads |
| Q10 | **b) Handles morphological variations and code-switching** | BPE can handle unseen Creole words by breaking them into known subword units |

---

## Lesson 02: LLM Fine-Tuning

| Question | Answer | Explanation |
|----------|--------|-------------|
| Q1 | **b) Pre-training learns general knowledge; fine-tuning adapts** | Pre-training on massive corpora gives general understanding; fine-tuning specializes for Caribbean context |
| Q2 | **a) Low-Rank Adaptation; reduces trainable parameters** | LoRA adds small rank-r matrices, training < 1% of total parameters |
| Q3 | **c) Less than 1%** | With r=16, d=4096: trainable = 2 * 4096 * 16 = 131K vs 4096^2 = 16.7M per layer (< 1%) |
| Q4 | **b) Foundation models underrepresent Caribbean culture** | Training data skews toward US/European English; Caribbean context is largely missing |
| Q5 | **b) Instruction format with system/user/assistant roles** | Chat models expect structured message format for proper instruction following |
| Q6 | **b) Simulate larger batch sizes when GPU memory is limited** | Accumulate gradients over multiple mini-batches before updating weights |
| Q7 | **b) Creole words split into many subword tokens** | Tokenizers trained on English may fragment Creole, increasing sequence length and reducing quality |
| Q8 | **b) Model loses general capabilities while adapting** | Aggressive fine-tuning can overwrite useful pre-trained knowledge; LoRA helps mitigate this |
| Q9 | **c) Combination of factual accuracy, cultural sensitivity, and human evaluation** | No single metric captures cultural understanding; human Caribbean evaluators are essential |
| Q10 | **b) Gradually increase LR to prevent destructive early updates** | Pretrained weights are delicate; sudden large gradients can destroy useful representations |

---

## Lesson 03: Diffusion Models

| Question | Answer | Explanation |
|----------|--------|-------------|
| Q1 | **b) Gradually adding Gaussian noise over T timesteps** | Forward process q(x_t|x_{t-1}) adds small amounts of noise until data becomes pure Gaussian noise |
| Q2 | **b) Predict and remove noise iteratively** | The reverse process learns p(x_{t-1}|x_t), generating data by iterative denoising |
| Q3 | **b) Allows jumping directly to any timestep** | Using alpha_bar_t, we compute x_t from x_0 directly without stepping through all intermediate states |
| Q4 | **b) The noise epsilon added at timestep t** | The network predicts epsilon; we can then compute x_0 and the posterior mean |
| Q5 | **b) Determines noise addition speed and denoising difficulty** | Too fast = hard to denoise; too slow = wasted computation |
| Q6 | **b) MSE between predicted and actual noise** | L = E[||epsilon - epsilon_theta(x_t, t)||^2] - simple and effective |
| Q7 | **b) Generating probabilistic ensemble forecasts** | Sample multiple possible future tracks from the learned distribution for uncertainty quantification |
| Q8 | **c) 1000** | Original DDPM uses T=1000; modern methods (DDIM) can reduce to 50-100 steps |
| Q9 | **b) Noise prediction implicitly learns the score function** | epsilon_theta relates to the score via: score = -epsilon / sqrt(1-alpha_bar), connecting DDPM to score-based models |
| Q10 | **b) Interpolates between conditional and unconditional generation** | CFG uses: output = (1+w)*conditional - w*unconditional, controlling quality vs diversity |

---

## Lesson 04: MLOps & Deployment

| Question | Answer | Explanation |
|----------|--------|-------------|
| Q1 | **b) Many rural areas have intermittent or no connectivity** | Caribbean farmers, fishers, and communities in remote areas cannot rely on cloud-only solutions |
| Q2 | **b) Reduces model size and inference time with INT8** | 4x smaller model, 2-4x faster inference, with typically < 1% accuracy loss |
| Q3 | **b) Service running, model loaded, can accept requests** | Health endpoints let load balancers and monitoring systems verify system readiness |
| Q4 | **b) Distribution of incoming data changes vs training data** | Model trained on dry-season crop images may fail during wet season if not monitored |
| Q5 | **b) Quick local inference offline, cloud refinement when connected** | Matches the intermittent connectivity reality across Caribbean islands |
| Q6 | **b) Track model versions, metrics, and manage promotion** | Like Git for models - track which version is in production, compare metrics, rollback if needed |
| Q7 | **b) Sustained drop in average confidence** | Indicates the model is encountering data it was not trained on (drift) |
| Q8 | **b) Lower network latency for Caribbean users** | Miami is ~50ms from Caribbean; US-East is ~80-100ms; US-West is ~150ms+ |
| Q9 | **c) .tflite or .onnx** | TFLite and ONNX are optimized for mobile/edge inference with small footprint |
| Q10 | **b) Limited labeled images of Caribbean-specific crop diseases** | Requires field partnerships with agricultural extension services across multiple islands |

---

## Lesson 05: RAG Systems

| Question | Answer | Explanation |
|----------|--------|-------------|
| Q1 | **b) Retrieval Augmented Generation; reduces hallucination** | RAG grounds LLM answers in retrieved factual documents rather than relying on potentially incorrect parametric memory |
| Q2 | **b) Word is rare across corpus, more discriminative** | Common words (the, is) have low IDF; rare domain terms (Kaieteur, calypso) have high IDF |
| Q3 | **b) Measures angle (direction), not magnitude** | A 100-word and 1000-word document about cricket should be similar; cosine ignores length |
| Q4 | **b) Split into smaller retrievable units** | Large documents may contain irrelevant sections; chunks allow precise retrieval |
| Q5 | **b) Losing context that spans chunk boundaries** | A legal clause split across chunks loses meaning; overlap preserves boundary context |
| Q6 | **b) Purely lexical, cannot capture semantic similarity** | TF-IDF cannot match "hurricane" with "tropical cyclone"; dense embeddings can |
| Q7 | **b) Fast approximate nearest neighbor search** | FAISS/Pinecone enables millisecond search over millions of vectors |
| Q8 | **b) Access up-to-date Caribbean knowledge without retraining** | New laws, statistics, events can be added to the knowledge base instantly |
| Q9 | **b) LLMs focus on beginning/end, ignoring middle** | Research shows attention drops for information in the middle of long contexts |
| Q10 | **b) Retrieval relevance, faithfulness, correctness, and human evaluation** | Multi-dimensional evaluation needed; automated metrics plus Caribbean human evaluators |

---

## Lesson 06: AI Entrepreneurship

| Question | Answer | Explanation |
|----------|--------|-------------|
| Q1 | **b) Different regulations, currencies, payment systems** | Each island is a sovereign nation with unique business environment; one-size-fits-all fails |
| Q2 | **b) Provides grants and financing for technology projects** | CDB has specific programs for technology and innovation supporting Caribbean development |
| Q3 | **b) Affordable SaaS ($10-20/month) with offline capabilities** | Smallholder farmers have limited budgets; affordable recurring pricing with offline access is key |
| Q4 | **b) Many target users in rural areas lack reliable internet** | Blue Mountain coffee farmers, Guyanese rice growers, Vincentian banana farmers need offline tools |
| Q5 | **b) Facilitates free movement of services and skilled persons** | CSME allows Caribbean businesses to operate across member states more easily |
| Q6 | **b) Ensuring model does not encode or amplify biases** | Caribbean societies are ethnically diverse; credit models must be fair across all groups |
| Q7 | **b) Access to talent, investment, and tech hub connections** | Caribbean diaspora in US, UK, Canada provides funding, expertise, and market access |
| Q8 | **c) Guyana** | Guyana's oil boom creates massive demand for industrial AI in extraction, monitoring, and logistics |
| Q9 | **b) Focus on ONE country first, then expand** | Prove product-market fit locally before navigating multi-country complexity |
| Q10 | **b) Prioritize Caribbean-based storage, comply with local laws** | Data sovereignty protects Caribbean interests; comply with Jamaica Data Protection Act, etc. |

---

## Lesson 07: Advanced Reinforcement Learning

| Question | Answer | Explanation |
|----------|--------|-------------|
| Q1 | **b) Clips policy ratio for training stability** | PPO's clipped objective prevents destructively large policy updates that destabilize training |
| Q2 | **b) How much better/worse an action is vs average** | A(s,a) = Q(s,a) - V(s); positive = better than average, negative = worse |
| Q3 | **b) Lower fuel costs, weather risk, plus efficiency bonus** | The reward function penalizes distance (fuel) and weather, while bonusing short routes |
| Q4 | **b) Smoothly trades off between bias and variance** | Lambda=0 gives high bias/low variance; lambda=1 gives low bias/high variance; GAE interpolates |
| Q5 | **b) Prevents agent from only helping one island** | Without equity constraint, agent could maximize reward by ignoring small, hard-to-reach islands |
| Q6 | **b) Maximum allowed change in probability ratios** | Clip at [1-0.2, 1+0.2] = [0.8, 1.2] means policy can change by at most 20% per update |
| Q7 | **b) Optimizes LLM to generate higher-scored responses** | PPO treats the LLM as a policy, reward model as the environment reward, optimizing for human preferences |
| Q8 | **b) Comprehensive situational awareness** | Port, cargo, demand, weather, and time all affect optimal routing decisions |
| Q9 | **b) Baseline to reduce variance in policy gradient estimates** | Subtracting V(s) from returns reduces variance without introducing bias |
| Q10 | **b) RL adapts to dynamic, uncertain conditions** | Static optimization assumes known parameters; RL learns policies that handle uncertainty inherent in disaster response |

---

> "If yuh get 80% or more, yuh ready fi di real world.
> If not, go back and study - no shame in dat!"
>
> - Adrian Dunkley, Caribbean AI Academy
