"""
=============================================================================
 CARIBBEAN AI ACADEMY - GRADUATES MODULE (05)
 Lesson 07: Advanced Reinforcement Learning - PPO for Caribbean Logistics
 By Adrian Dunkley
=============================================================================

 Alright Graduates, last lesson before yuh capstone projects!
 We tackling Proximal Policy Optimization (PPO) - di algorithm
 behind ChatGPT's RLHF and many real-world RL deployments.

 But we nah just learn theory - we apply it to REAL Caribbean
 problems: inter-island shipping optimization and hurricane
 resource distribution. These are life-and-death problems
 fi island nations, and RL can help solve dem.

 From Kingston harbor to Port of Spain port, from Nassau to
 Georgetown - logistics is di lifeblood of Caribbean trade.

 LEARNING OBJECTIVES:
 1. Understand PPO algorithm and why it works
 2. Implement policy gradient concepts from scratch
 3. Apply to Caribbean inter-island shipping optimization
 4. Build hurricane resource distribution simulator
 5. Grasp RLHF connection (how ChatGPT uses PPO)

 # 🧩 Puzzle Piece 10/12: The 'ranger' is colored like limes, grass, and the go signal
=============================================================================
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

# =============================================================================
# PART 1: POLICY GRADIENT FOUNDATIONS
# =============================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class PolicyNetwork:
    """
    Simple policy network dat maps states to action probabilities.

    In RL, di policy pi(a|s) tells di agent what action fi take
    in each state. Unlike Q-learning (which learns values),
    policy gradient methods directly optimize di policy.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Two-layer network
        np.random.seed(42)
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(action_dim)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Compute action probabilities given state."""
        h = np.maximum(0, state @ self.W1 + self.b1)  # ReLU
        logits = h @ self.W2 + self.b2
        probs = softmax(logits)
        return probs

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Sample action from policy and return log probability."""
        probs = self.forward(state)
        action = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        return action, log_prob


class ValueNetwork:
    """
    Value network V(s) estimates expected return from state s.
    Used as baseline in PPO to reduce variance.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        np.random.seed(43)
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)

    def forward(self, state: np.ndarray) -> float:
        """Estimate state value."""
        h = np.maximum(0, state @ self.W1 + self.b1)
        value = (h @ self.W2 + self.b2)[0]
        return value


# =============================================================================
# PART 2: PPO ALGORITHM CONCEPTS
# =============================================================================

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.

    PPO is di most popular RL algorithm because:
    1. It's stable (small policy updates)
    2. It's sample efficient
    3. It works well across many problems
    4. OpenAI uses it fi RLHF in ChatGPT

    Di key idea: CLIP di policy ratio to prevent too-large updates.

    L_CLIP = min(r(theta) * A, clip(r(theta), 1-eps, 1+eps) * A)

    Where:
    - r(theta) = pi_new(a|s) / pi_old(a|s)  (probability ratio)
    - A = advantage estimate (how much better dis action was than average)
    - eps = clip range (typically 0.2)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 clip_epsilon: float = 0.2, gamma: float = 0.99,
                 lam: float = 0.95):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lam = lam  # GAE lambda

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def act(self, state: np.ndarray) -> int:
        """Select action and store experience."""
        action, log_prob = self.policy.select_action(state)
        value = self.value.forward(state)

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

        return action

    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation (GAE).

        Advantage tells us: "Was dis action BETTER or WORSE than average?"
        A(s, a) = Q(s, a) - V(s)

        GAE smoothly interpolates between:
        - lambda=0: A = r + gamma*V(s') - V(s)  (high bias, low variance)
        - lambda=1: A = sum of discounted rewards - V(s) (low bias, high variance)
        """
        advantages = np.zeros(len(self.rewards))
        returns = np.zeros(len(self.rewards))
        gae = 0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def compute_ppo_loss(self, old_log_probs: np.ndarray,
                          new_log_probs: np.ndarray,
                          advantages: np.ndarray) -> float:
        """
        Compute PPO clipped objective.

        Dis is di heart of PPO - it prevents di policy from changing
        too much in one update, which could destabilize training.
        """
        # Probability ratio
        ratio = np.exp(new_log_probs - old_log_probs)

        # Clipped ratio
        clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # PPO objective: min of unclipped and clipped
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages
        ppo_loss = -np.mean(np.minimum(loss1, loss2))

        return ppo_loss

    def clear_buffer(self):
        """Clear experience buffer after update."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()


# =============================================================================
# PART 3: CARIBBEAN SHIPPING ENVIRONMENT
# =============================================================================

class CaribbeanShippingEnv:
    """
    Inter-island shipping optimization environment.

    Agent must decide how fi route cargo between Caribbean ports
    considering: distance, demand, weather, port capacity.

    Ports: Kingston (JA), Port of Spain (TT), Bridgetown (BB),
           Nassau (BS), Georgetown (GY), Castries (LC)
    """

    PORTS = {
        0: {"name": "Kingston", "country": "Jamaica", "capacity": 100},
        1: {"name": "Port of Spain", "country": "Trinidad", "capacity": 80},
        2: {"name": "Bridgetown", "country": "Barbados", "capacity": 60},
        3: {"name": "Nassau", "country": "Bahamas", "capacity": 70},
        4: {"name": "Georgetown", "country": "Guyana", "capacity": 50},
        5: {"name": "Castries", "country": "St. Lucia", "capacity": 40},
    }

    # Distance matrix (simplified, in nautical miles)
    DISTANCES = np.array([
        [0,   1100, 1300, 500,  1500, 1200],   # Kingston
        [1100, 0,    250,  1400, 350,  200],    # Port of Spain
        [1300, 250,  0,    1500, 500,  150],    # Bridgetown
        [500,  1400, 1500, 0,    2000, 1400],   # Nassau
        [1500, 350,  500,  2000, 0,    450],    # Georgetown
        [1200, 200,  150,  1400, 450,  0],      # Castries
    ])

    def __init__(self, num_ports: int = 6):
        self.num_ports = num_ports
        self.current_port = 0
        self.cargo = 0
        self.demand = np.zeros(num_ports)
        self.weather_risk = np.zeros(num_ports)
        self.step_count = 0
        self.max_steps = 20
        self.total_revenue = 0

    def reset(self) -> np.ndarray:
        """Reset environment fi new episode."""
        self.current_port = np.random.randint(self.num_ports)
        self.cargo = np.random.randint(20, 80)
        self.demand = np.random.uniform(10, 60, self.num_ports)
        self.weather_risk = np.random.uniform(0, 0.5, self.num_ports)
        self.step_count = 0
        self.total_revenue = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        State vector:
        - Current port (one-hot, 6 dims)
        - Cargo amount (1 dim)
        - Demand at each port (6 dims)
        - Weather risk at each port (6 dims)
        - Steps remaining (1 dim)
        """
        port_onehot = np.zeros(self.num_ports)
        port_onehot[self.current_port] = 1

        state = np.concatenate([
            port_onehot,
            [self.cargo / 100],
            self.demand / 60,
            self.weather_risk,
            [(self.max_steps - self.step_count) / self.max_steps]
        ])
        return state

    @property
    def state_dim(self) -> int:
        return self.num_ports * 3 + 2  # one-hot + cargo + demand + weather + steps

    @property
    def action_dim(self) -> int:
        return self.num_ports  # Choose which port to go to

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action: sail to chosen port and deliver cargo.

        Reward considers:
        - Revenue from cargo delivery (demand * amount)
        - Fuel cost (proportional to distance)
        - Weather penalty (risk of delay/damage)
        - Port congestion
        """
        destination = action
        distance = self.DISTANCES[self.current_port, destination]

        # Fuel cost (proportional to distance)
        fuel_cost = distance * 0.1

        # Weather penalty
        weather_penalty = self.weather_risk[destination] * 20

        # Revenue from delivery
        delivery_amount = min(self.cargo, self.demand[destination])
        revenue = delivery_amount * 2.0  # $2 per unit of cargo

        # Update demand
        self.demand[destination] = max(0, self.demand[destination] - delivery_amount)

        # Calculate reward
        reward = revenue - fuel_cost - weather_penalty

        # Bonus for efficient routing (staying in area)
        if distance < 400:
            reward += 5  # Bonus fi short routes

        # Update state
        self.current_port = destination
        self.cargo = max(0, self.cargo - delivery_amount)
        self.cargo += np.random.randint(0, 20)  # Pick up new cargo
        self.total_revenue += reward
        self.step_count += 1

        # Refresh weather
        self.weather_risk = np.clip(
            self.weather_risk + np.random.normal(0, 0.05, self.num_ports),
            0, 1
        )

        done = self.step_count >= self.max_steps
        info = {
            "port": self.PORTS[destination]["name"],
            "country": self.PORTS[destination]["country"],
            "distance": distance,
            "revenue": revenue,
            "fuel_cost": fuel_cost,
            "total_revenue": self.total_revenue,
        }

        return self._get_state(), reward, done, info


# =============================================================================
# PART 4: HURRICANE RESOURCE DISTRIBUTION
# =============================================================================

class HurricaneResourceEnv:
    """
    Hurricane resource distribution environment.

    After a hurricane, limited resources (food, water, medical supplies,
    tarps, generators) must be distributed across affected islands.
    Di agent must decide WHERE to send resources based on:
    - Damage severity
    - Population affected
    - Current supply levels
    - Transportation accessibility

    Dis is a real problem after every hurricane season.
    """

    ISLANDS = {
        0: {"name": "Dominica", "population": 72000, "vulnerability": 0.9},
        1: {"name": "Barbuda", "population": 1800, "vulnerability": 0.95},
        2: {"name": "St. Martin", "population": 40000, "vulnerability": 0.7},
        3: {"name": "Puerto Rico", "population": 3200000, "vulnerability": 0.6},
        4: {"name": "US Virgin Islands", "population": 105000, "vulnerability": 0.7},
        5: {"name": "Bahamas (Abaco)", "population": 17000, "vulnerability": 0.85},
    }

    RESOURCE_TYPES = ["water", "food", "medical", "tarps", "generators"]

    def __init__(self):
        self.num_islands = len(self.ISLANDS)
        self.num_resources = len(self.RESOURCE_TYPES)
        self.damage_levels = np.zeros(self.num_islands)
        self.supply_levels = np.zeros((self.num_islands, self.num_resources))
        self.available_resources = np.zeros(self.num_resources)
        self.day = 0
        self.max_days = 14  # Two weeks of relief operations

    def reset(self) -> np.ndarray:
        """Reset after hurricane strike."""
        # Simulate hurricane damage
        self.damage_levels = np.array([
            island["vulnerability"] * np.random.uniform(0.5, 1.0)
            for island in self.ISLANDS.values()
        ])

        # Initial supply levels (depleted by hurricane)
        self.supply_levels = np.random.uniform(0, 0.3, (self.num_islands, self.num_resources))

        # Available resources at distribution hub
        self.available_resources = np.array([500, 400, 200, 300, 50], dtype=float)

        self.day = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """State: damage + supplies + available resources + day."""
        return np.concatenate([
            self.damage_levels,
            self.supply_levels.flatten(),
            self.available_resources / 500,
            [self.day / self.max_days]
        ])

    @property
    def state_dim(self) -> int:
        return self.num_islands + self.num_islands * self.num_resources + self.num_resources + 1

    @property
    def action_dim(self) -> int:
        return self.num_islands  # Which island to prioritize

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Send resources to chosen island.
        Reward based on lives saved and suffering reduced.
        """
        target = action

        # Send a portion of available resources
        send_ratio = 0.3  # Send 30% of available to target
        sent = self.available_resources * send_ratio

        # Update supply levels
        population_factor = list(self.ISLANDS.values())[target]["population"] / 3200000
        supply_increase = sent / (1000 * max(0.1, population_factor))
        self.supply_levels[target] = np.clip(
            self.supply_levels[target] + supply_increase, 0, 1
        )

        # Decrease available resources
        self.available_resources = np.clip(self.available_resources - sent, 0, None)

        # Resupply (new aid arrives each day)
        self.available_resources += np.array([100, 80, 40, 60, 10]) * np.random.uniform(0.5, 1.5)

        # Compute reward
        # Higher reward for helping highly damaged islands with low supplies
        need_score = self.damage_levels[target] * (1 - self.supply_levels[target].mean())
        equity_bonus = -np.std(self.supply_levels.mean(axis=1)) * 10  # Penalize inequality
        urgency_factor = 1 + (self.damage_levels[target] > 0.8) * 0.5

        reward = need_score * urgency_factor * 10 + equity_bonus

        # Natural recovery (supplies get consumed, slow rebuilding)
        self.supply_levels *= 0.95  # Daily consumption
        self.damage_levels *= 0.98  # Slow recovery

        self.day += 1
        done = self.day >= self.max_days

        info = {
            "island": self.ISLANDS[target]["name"],
            "damage": self.damage_levels[target],
            "avg_supply": self.supply_levels[target].mean(),
            "equity": 1 - np.std(self.supply_levels.mean(axis=1)),
        }

        return self._get_state(), reward, done, info


# =============================================================================
# PART 5: TRAINING SIMULATION
# =============================================================================

def train_shipping_agent(num_episodes: int = 100):
    """Train PPO agent on Caribbean shipping optimization."""
    print("=" * 60)
    print(" TRAINING: CARIBBEAN SHIPPING OPTIMIZER")
    print("=" * 60)

    env = CaribbeanShippingEnv()
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    episode_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store_reward(reward, done)
            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

        # Compute advantages and clear buffer
        if len(agent.rewards) > 0:
            advantages, returns = agent.compute_gae()
            agent.clear_buffer()

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"  Episode {episode + 1:>4}: avg_reward = {avg_reward:>8.2f} | "
                  f"last_port = {info['port']}")

    print(f"\nFinal avg reward (last 20): {np.mean(episode_rewards[-20:]):.2f}")
    return episode_rewards


def train_hurricane_agent(num_episodes: int = 100):
    """Train PPO agent on hurricane resource distribution."""
    print("\n" + "=" * 60)
    print(" TRAINING: HURRICANE RESOURCE DISTRIBUTION")
    print("=" * 60)

    env = HurricaneResourceEnv()
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    episode_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for day in range(env.max_days):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store_reward(reward, done)
            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

        if len(agent.rewards) > 0:
            advantages, returns = agent.compute_gae()
            agent.clear_buffer()

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"  Episode {episode + 1:>4}: avg_reward = {avg_reward:>8.2f} | "
                  f"equity = {info['equity']:.3f} | last_island = {info['island']}")

    print(f"\nFinal avg reward (last 20): {np.mean(episode_rewards[-20:]):.2f}")
    return episode_rewards


# =============================================================================
# PART 6: RLHF CONNECTION
# =============================================================================

def explain_rlhf_connection():
    """Explain how PPO connects to RLHF in LLMs like ChatGPT."""
    print("\n" + "=" * 60)
    print(" PPO IN RLHF: HOW CHATGPT LEARNS FROM HUMANS")
    print("=" * 60)

    explanation = """
    RLHF (Reinforcement Learning from Human Feedback) uses PPO:

    Step 1: Supervised Fine-Tuning (SFT)
    - Train LLM on high-quality demonstrations
    - Like teaching a child: "Dis is how yuh answer questions properly"

    Step 2: Reward Model Training
    - Humans rank multiple LLM responses (best to worst)
    - Train a reward model to predict human preferences
    - Like training a cricket umpire to judge good bowling

    Step 3: PPO Optimization
    - LLM generates responses (POLICY)
    - Reward model scores dem (REWARD)
    - PPO updates di LLM to generate higher-scored responses
    - Clip ratio prevents di LLM from changing too drastically

    Caribbean Analogy:
    - SFT = Learning di basics of cooking from grandma
    - Reward Model = Learning what "good" jerk chicken taste like
    - PPO = Practicing until yuh jerk chicken consistently score high

    Why PPO for RLHF?
    1. Stability: Won't destroy the base model's knowledge
    2. The clip mechanism prevents reward hacking
    3. Works well with large models and distributed training
    4. Proven track record across different applications
    """
    print(explanation)


# =============================================================================
# PART 7: QUIZ
# =============================================================================

QUIZ_QUESTIONS = """
=============================================================================
 QUIZ: ADVANCED REINFORCEMENT LEARNING (10 Questions)
=============================================================================

Q1: What is the key innovation of PPO compared to basic policy gradient?
    a) It uses a larger neural network
    b) It clips the policy ratio to prevent destructively large updates,
       ensuring training stability
    c) It uses a different reward function
    d) It trains faster on GPUs

Q2: What is the advantage function A(s, a) in RL?
    a) The total reward
    b) How much BETTER (or worse) an action is compared to the average
       action in that state: A(s,a) = Q(s,a) - V(s)
    c) The probability of winning
    d) The learning rate

Q3: In the Caribbean shipping environment, why might the agent learn
    to prefer short-distance routes?
    a) Short routes are always more profitable
    b) Lower fuel costs and weather risk, plus the efficiency bonus,
       make short routes more rewarding on average
    c) The agent cannot learn about distances
    d) Short routes are faster to compute

Q4: What is GAE (Generalized Advantage Estimation) and why use it?
    a) A type of neural network
    b) A method that smoothly trades off between bias and variance
       in advantage estimation using lambda parameter
    c) A data augmentation technique
    d) A reward normalization method

Q5: In hurricane resource distribution, why is an equity bonus important?
    a) To make the math simpler
    b) To prevent the agent from repeatedly helping only one island
       while others suffer, ensuring fair distribution across all
       affected communities
    c) To reduce computation
    d) To speed up training

Q6: What does the clip epsilon (typically 0.2) in PPO control?
    a) The learning rate
    b) The maximum allowed change in policy probability ratios per
       update, preventing the policy from changing too drastically
    c) The discount factor
    d) The batch size

Q7: How is PPO used in RLHF for training language models like ChatGPT?
    a) It replaces the transformer architecture
    b) It optimizes the LLM policy to generate responses that score
       higher on a reward model trained on human preferences
    c) It adds more training data
    d) It compresses the model

Q8: For Caribbean inter-island logistics, what state information is
    most critical for the RL agent?
    a) Only the current port location
    b) Current port, cargo levels, demand at each port, weather risk,
       and remaining time - comprehensive situational awareness
    c) Only the weather
    d) Only the distance matrix

Q9: What is the role of the value function V(s) in PPO?
    a) It selects actions
    b) It serves as a baseline to reduce variance in policy gradient
       estimates, making training more stable
    c) It stores experiences
    d) It normalizes rewards

Q10: Why is RL particularly well-suited for Caribbean disaster response
     compared to traditional optimization?
    a) RL is always better than optimization
    b) RL can adapt to dynamic, uncertain conditions (changing weather,
       infrastructure damage, supply chain disruptions) by learning
       policies that handle uncertainty, unlike static optimization
    c) RL requires less data
    d) RL is faster to compute
"""


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" CARIBBEAN AI ACADEMY - GRADUATES MODULE")
    print(" Lesson 07: Advanced RL - PPO for Caribbean Logistics")
    print(" By Adrian Dunkley")
    print("=" * 60)

    # Train shipping agent
    shipping_rewards = train_shipping_agent(num_episodes=100)

    # Train hurricane agent
    hurricane_rewards = train_hurricane_agent(num_episodes=100)

    # RLHF explanation
    explain_rlhf_connection()

    # Quiz
    print(QUIZ_QUESTIONS)

    print("\n" + "=" * 60)
    print(" Yuh now understand PPO - di algorithm behind ChatGPT!")
    print(" From Caribbean shipping to hurricane response,")
    print(" RL can optimize complex real-world decisions.")
    print(" Time fi yuh capstone projects! Show di world what")
    print(" Caribbean AI graduates can build!")
    print("=" * 60)

# 🧩 Puzzle Piece 10/12: The 'ranger' is colored like limes, grass, and the go signal
