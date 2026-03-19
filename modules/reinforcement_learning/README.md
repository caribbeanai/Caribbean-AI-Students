# Reinforcement Learning - Caribbean AI Curriculum

**Designed by Adrian Dunkley ([Adriandunkley.net](https://Adriandunkley.net)) | FREE Curriculum**

---

## Welcome to Reinforcement Learning!

Alright, Caribbean learner — yuh done learn supervised and unsupervised learning. Now we tackling something different: **Reinforcement Learning (RL)**. This is how machines learn by trial and error, just like how a young sailor learns to navigate Caribbean waters — by trying, failing, and getting better each time.

---

## Table of Contents

1. [What is Reinforcement Learning?](#what-is-reinforcement-learning)
2. [Key Concepts](#key-concepts)
3. [The Math Behind RL](#the-math-behind-rl)
4. [Core Algorithms](#core-algorithms)
5. [Exploration vs. Exploitation](#exploration-vs-exploitation)
6. [Caribbean Applications](#caribbean-applications)
7. [Deep Reinforcement Learning](#deep-reinforcement-learning)
8. [Practical Guidance](#practical-guidance)
9. [Quiz](#quiz)

---

## What is Reinforcement Learning?

### The Young Sailor Analogy

Imagine a young sailor from Tobago learning to navigate the Caribbean Sea:

- **The sailor** = the **agent** (the learner)
- **The Caribbean Sea** = the **environment** (the world)
- **Current position, wind, waves** = the **state** (what the agent observes)
- **Turn left, go straight, adjust sails** = **actions** (what the agent can do)
- **Reaching the destination safely** = **positive reward** (+)
- **Running aground on a reef** = **negative reward** (-)
- **The accumulated experience** = the **policy** (the strategy for choosing actions)

The sailor doesn't have a textbook. They learn from experience: "Last time I sailed too close to that reef near Grenada, mi nearly wreck. This time, mi give it more room."

### How RL Differs from Other ML

| Aspect | Supervised | Unsupervised | Reinforcement |
|--------|-----------|-------------|---------------|
| **Data** | Labeled examples | Unlabeled data | Rewards from interaction |
| **Learns from** | Correct answers | Patterns | Trial and error |
| **Feedback** | Immediate, per example | None | Delayed, sparse |
| **Goal** | Predict accurately | Find structure | Maximize cumulative reward |
| **Caribbean analogy** | CXC exam with answer key | Exploring reef without guide | Learning to fish by doing |

---

## Key Concepts

### 1. Agent and Environment

```
    ┌─────────────────┐
    │   ENVIRONMENT    │
    │  (Caribbean Sea) │
    │                  │
    │  State: position,│──── State ────►┌─────────┐
    │  wind, waves     │               │  AGENT   │
    │                  │◄── Action ────│ (Sailor) │
    │  Reward: +/-     │──── Reward ──►│          │
    └─────────────────┘               └─────────┘
```

The agent observes the state, takes an action, receives a reward, and transitions to a new state. This cycle repeats.

### 2. States

The state is everything the agent needs to make a decision. For our sailor:
- Current GPS coordinates
- Wind speed and direction
- Wave height
- Distance to destination
- Fuel remaining
- Nearby obstacles (reefs, islands)

### 3. Actions

What the agent can do at each state:
- Turn port (left) or starboard (right)
- Increase or decrease speed
- Adjust sail angle
- Drop anchor
- Change heading by X degrees

### 4. Rewards

The signal that tells the agent how well it did:

| Event | Reward |
|-------|--------|
| Move closer to destination | +1 |
| Arrive safely at port | +100 |
| Hit a reef | -50 |
| Run out of fuel | -30 |
| Each time step (to encourage speed) | -0.1 |
| Navigate through tricky passage | +20 |

### 5. Policy (pi)

The policy is the agent's strategy — a mapping from states to actions.

- **Deterministic policy:** In state S, always do action A
- **Stochastic policy:** In state S, do action A with probability P

The goal is to find the **optimal policy** — the strategy that maximizes total reward over time.

### 6. Value Function

How good is it to be in a particular state?

- **V(s):** The expected total future reward starting from state s
- Example: Being 1km from port with clear waters → high value. Being near a reef with low fuel → low value.

### 7. Q-Function (Action-Value)

How good is it to take a particular action in a particular state?

- **Q(s, a):** Expected total future reward of taking action a in state s, then following the optimal policy
- Example: Q(near_reef, turn_away) >> Q(near_reef, go_straight)

### 8. Discount Factor (gamma)

How much do we value future rewards vs. immediate rewards?

- **gamma = 0:** Only care about immediate reward (short-sighted)
- **gamma = 1:** Value future rewards equally (long-sighted)
- **gamma = 0.9:** Typical — value future rewards but not as much as immediate

Caribbean analogy: Would yuh rather have $100 JMD today or $110 JMD next week? The discount factor captures this time preference.

---

## The Math Behind RL

### The Bellman Equation

The foundation of RL. It says: the value of a state equals the immediate reward plus the discounted value of the next state.

```
V(s) = max_a [R(s,a) + gamma * V(s')]

Where:
  V(s)  = value of current state
  R(s,a) = immediate reward for action a in state s
  gamma  = discount factor (0 to 1)
  V(s') = value of the next state
  max_a  = choose the action that gives the best result
```

**Intuition:** The value of being in Kingston harbour = the best immediate reward you can get + the discounted value of wherever you end up next.

### Q-Learning Update Rule

```
Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]

Where:
  alpha  = learning rate (how quickly we update beliefs)
  R      = received reward
  gamma  = discount factor
  max_a' Q(s',a') = best Q-value in the next state
  [R + gamma * max_a' Q(s',a') - Q(s,a)] = TD error (the "surprise")
```

**Intuition:** If the outcome was better than expected (positive TD error), increase Q. If worse, decrease Q. Like the sailor adjusting: "That route was better than I thought — I'll take it more often."

### The Return

Total discounted reward from time t:

```
G_t = R_t + gamma * R_{t+1} + gamma^2 * R_{t+2} + ...

Example (sailor's journey):
  t=0: Reward +1 (moved toward port)
  t=1: Reward +1
  t=2: Reward -5 (hit rough waters)
  t=3: Reward +100 (arrived!)

  With gamma=0.9:
  G_0 = 1 + 0.9(1) + 0.81(-5) + 0.729(100) = 1 + 0.9 - 4.05 + 72.9 = 70.75
```

---

## Core Algorithms

### 1. Q-Learning (Off-Policy, Model-Free)

The workhorse of tabular RL. Learns the optimal Q-function directly.

```
Algorithm:
1. Initialize Q-table with zeros
2. For each episode:
   a. Start in initial state
   b. Choose action (epsilon-greedy)
   c. Take action, observe reward and new state
   d. Update Q(s,a) using Bellman equation
   e. Move to new state
   f. Repeat until done
```

**Caribbean example:** The sailor starts with no knowledge (Q-table all zeros). After many voyages (episodes), they learn which actions work best in each situation.

### 2. SARSA (On-Policy, Model-Free)

Like Q-Learning, but updates using the actual next action (not the best possible).

```
Q(s,a) <- Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]

Key difference: Uses Q(s',a') instead of max_a' Q(s',a')
  - Q-Learning: "What if I did the BEST thing next?"
  - SARSA: "What did I ACTUALLY do next?"
```

**Caribbean analogy:** Q-Learning is an optimist — "If I reach Barbados, I'll take the best route." SARSA is a realist — "If I reach Barbados, I'll probably take the route I usually take."

### 3. Monte Carlo Methods

Learn from complete episodes — play the whole game, then update.

```
1. Play a full episode (sail from A to B)
2. For each state visited, calculate the actual return G
3. Update V(s) toward the observed G
```

**Caribbean analogy:** Like reviewing a full cricket innings — you wait until it's done, then analyze what worked and what didn't.

### 4. Policy Gradient Methods

Instead of learning value functions, directly optimize the policy.

```
Objective: Maximize expected return
Update: theta <- theta + alpha * gradient(J(theta))

Where theta are the policy parameters.
```

**Caribbean analogy:** Instead of learning "how valuable is each position on the cricket pitch," directly learn "what shots to play" — optimize your batting strategy.

---

## Exploration vs. Exploitation

### The Fisherman's Dilemma

A Bahamian fisherman faces a choice every morning:
- **Exploit:** Go to the fishing spot that has been reliable (known good spot)
- **Explore:** Try a new spot that might be even better (or might be empty)

Too much exploitation → miss discovering better spots
Too much exploration → waste time on bad spots when you know good ones

### Epsilon-Greedy Strategy

```
With probability (1-epsilon): Choose the best known action (exploit)
With probability epsilon:     Choose a random action (explore)

Common: Start with epsilon=1.0 (all exploration)
        Gradually decrease to epsilon=0.01 (mostly exploitation)
```

Think of it like a new arrival to Jamaica: at first, yuh try every restaurant (explore). Over time, yuh mostly go to yuh favourites (exploit) but occasionally try somewhere new.

### Other Strategies

- **Boltzmann/Softmax:** Choose actions proportional to their estimated value
- **UCB (Upper Confidence Bound):** Prefer actions with high uncertainty
- **Thompson Sampling:** Bayesian approach — sample from belief distributions

---

## Caribbean Applications

### 1. Navigation and Maritime

**Training a Virtual Sailor:**
- Learn optimal routes between Caribbean islands
- Account for weather patterns, currents, fuel costs
- Avoid hazards (reefs, shallow waters, shipping lanes)
- Adapt to changing conditions in real time

**Port Operations:**
- Optimize ship docking sequences at Kingston Harbour
- Manage container yard operations
- Coordinate tugboat assignments

### 2. Transportation

**Optimizing Caribbean Bus Routes:**
- Learn bus schedules that minimize wait times
- Adapt to traffic patterns in Kingston, Port of Spain, Bridgetown
- Handle route disruptions (road closures, events)
- Balance coverage of rural and urban areas

**Taxi/Ride-sharing:**
- Where should drivers position themselves?
- Dynamic pricing during peak times (cruise ship arrivals)
- Route optimization through Caribbean city traffic

### 3. Tourism and Hospitality

**Managing a Virtual Caribbean Hotel:**
- Dynamic room pricing (peak season vs. off-season)
- Staff scheduling based on occupancy predictions
- Resource allocation (pool maintenance, restaurant staffing)
- Energy management (AC during hot Caribbean days)

**Tour Planning:**
- Create optimal island-hopping itineraries
- Balance popular and off-the-beaten-path attractions
- Adapt to weather changes mid-trip

### 4. Energy Management

**Caribbean Renewable Energy Grid:**
- Balance solar and wind generation with demand
- Decide when to store energy vs. use immediately
- Manage the transition from fossil fuels
- Handle the intermittency of tropical weather

### 5. Agriculture

**Smart Farming Decisions:**
- When to irrigate crops based on weather forecasts
- Optimal harvesting timing for sugarcane and bananas
- Pest management strategies
- Crop rotation planning

### 6. Cricket Strategy

**West Indies Cricket AI:**
- Optimal batting order decisions
- Bowling changes and field placement
- DLS method scenario planning
- CPL auction bid strategies

---

## Deep Reinforcement Learning

When states are too numerous to fit in a table, we use **neural networks** to approximate value functions or policies.

### DQN (Deep Q-Network)

Replace the Q-table with a neural network:
- Input: State features
- Output: Q-value for each action
- Train using experience replay and target networks

**Caribbean analogy:** Instead of memorizing every possible sailing scenario, the sailor develops "intuition" (the neural network) that generalizes to new situations.

### Policy Gradient / Actor-Critic

- **Actor:** The policy network (decides actions)
- **Critic:** The value network (evaluates how good the action was)

Like a cricket partnership — the batsman (actor) plays shots, and the partner (critic) calls "yes" or "no" for the run.

### Popular Algorithms

| Algorithm | Type | Best For |
|-----------|------|----------|
| DQN | Value-based | Discrete actions (turn left/right) |
| A2C/A3C | Actor-Critic | Both discrete and continuous |
| PPO | Policy gradient | Stable training, general purpose |
| SAC | Actor-Critic | Continuous actions (steering angle) |
| TD3 | Actor-Critic | Continuous control |

---

## Practical Guidance

### Getting Started with RL

1. **Start simple:** Grid worlds and small problems
2. **Use Gymnasium (formerly OpenAI Gym):** Standard RL environment library
3. **Implement Q-Learning by hand:** Understand the fundamentals
4. **Then try deep RL:** Use Stable-Baselines3 for state-of-the-art algorithms

### Common Pitfalls

- **Reward shaping is tricky:** Bad rewards lead to unexpected behaviour. If you reward the sailor for speed, they might crash into port!
- **Training instability:** Deep RL can be unstable. Use target networks, gradient clipping.
- **Sample efficiency:** RL often needs millions of interactions. Start with simulations.
- **Exploration collapse:** If the agent stops exploring too early, it misses better strategies.

### Caribbean-Specific Considerations

- **Simulation first:** Build Caribbean simulations before deploying real systems
- **Safety constraints:** Maritime navigation requires hard safety constraints
- **Multi-objective:** Caribbean problems often have multiple goals (profit, sustainability, equity)
- **Small data:** Consider model-based RL when environment interactions are expensive

---

## Quiz

### Question 1
In reinforcement learning, what is the "agent"?

- A) The training data
- B) The learner that takes actions in an environment
- C) The reward function
- D) The environment itself

<details>
<summary>Answer</summary>
**B)** The agent is the learner that observes states, takes actions, and receives rewards. In our analogy, the agent is the sailor navigating Caribbean waters.
</details>

### Question 2
A Caribbean fisherman always goes to the same fishing spot. What RL concept does this illustrate?

- A) Exploration
- B) Exploitation
- C) Reward shaping
- D) Policy gradient

<details>
<summary>Answer</summary>
**B)** Exploitation means always choosing the action with the highest known reward. The fisherman exploits his knowledge of a good spot instead of exploring new ones.
</details>

### Question 3
What does the discount factor (gamma) control?

- A) The learning rate
- B) How much future rewards are valued relative to immediate rewards
- C) The number of episodes
- D) The exploration rate

<details>
<summary>Answer</summary>
**B)** Gamma (between 0 and 1) determines how much the agent values future rewards compared to immediate rewards. A gamma of 0.9 means future rewards are valued but less than immediate ones.
</details>

### Question 4
In Q-Learning, what does Q(s, a) represent?

- A) The probability of being in state s
- B) The expected total future reward of taking action a in state s
- C) The number of times action a was taken
- D) The immediate reward in state s

<details>
<summary>Answer</summary>
**B)** Q(s, a) is the expected cumulative discounted reward of taking action a in state s, then following the optimal policy thereafter.
</details>

### Question 5
Why is the exploration-exploitation tradeoff important?

- A) It affects the speed of computation
- B) Too much exploitation misses better strategies; too much exploration wastes time on suboptimal actions
- C) It only matters in supervised learning
- D) It determines the number of features

<details>
<summary>Answer</summary>
**B)** Balancing exploration (trying new things) and exploitation (using known good strategies) is crucial. The right balance changes over time — explore more early, exploit more later.
</details>

### Question 6
Which RL algorithm learns from complete episodes rather than step by step?

- A) Q-Learning
- B) SARSA
- C) Monte Carlo methods
- D) DQN

<details>
<summary>Answer</summary>
**C)** Monte Carlo methods wait until an episode is complete, then use the actual returns experienced to update value estimates. Like reviewing a full cricket innings after it ends.
</details>

### Question 7
A Caribbean hotel AI sets room prices. Prices are adjusted daily. The "state" would likely include:

- A) Only the current price
- B) Current occupancy, season, day of week, upcoming events, competitor prices
- C) Only the profit from yesterday
- D) The hotel's name

<details>
<summary>Answer</summary>
**B)** The state should include all relevant information for making pricing decisions: occupancy rates, seasonality, day of week, upcoming events, and competitor pricing.
</details>

### Question 8
What is the Bellman Equation used for?

- A) Calculating gradients in neural networks
- B) Expressing the relationship between the value of a state and the values of successor states
- C) Measuring classification accuracy
- D) Clustering data points

<details>
<summary>Answer</summary>
**B)** The Bellman Equation is the foundational recursion in RL: V(s) = max_a [R(s,a) + gamma * V(s')]. It says the value of a state equals the best immediate reward plus the discounted value of the next state.
</details>

### Question 9
Why might deep RL (using neural networks) be necessary for optimizing Caribbean bus routes?

- A) Because buses are large vehicles
- B) Because the state space (all possible combinations of bus positions, passenger locations, traffic) is too large for a table
- C) Because neural networks are always better
- D) Because Caribbean roads are complicated

<details>
<summary>Answer</summary>
**B)** The state space for bus route optimization is enormous — combinations of bus locations, passenger demand at each stop, traffic conditions, time of day, etc. A Q-table cannot represent this, so neural networks approximate the value function.
</details>

### Question 10
An RL agent trained to manage a Caribbean power grid gives a negative reward for blackouts and a positive reward for meeting demand with renewable energy. What might go wrong if the reward is poorly designed?

- A) Nothing — RL always finds the optimal solution
- B) The agent might find loopholes, such as shutting down parts of the grid to avoid "blackouts" in those areas while technically meeting its reward criteria
- C) The agent will refuse to train
- D) The grid will physically break

<details>
<summary>Answer</summary>
**B)** Reward hacking is a real RL problem. A poorly designed reward function can lead the agent to find unintended shortcuts that technically maximize reward but don't achieve the desired behaviour. Careful reward design is critical.
</details>

---

## Next Steps

- Try the `examples.py` file for hands-on Caribbean RL code
- Start with the grid world — understand Q-learning fundamentals
- Then try the taxi problem and resource management game
- Move on to **Large Language Models** for the latest in AI

---

*Designed by Adrian Dunkley ([Adriandunkley.net](https://Adriandunkley.net)) | FREE Caribbean AI Curriculum*
*Navigate yuh learning journey like a Caribbean captain — steady hand, open mind!*
