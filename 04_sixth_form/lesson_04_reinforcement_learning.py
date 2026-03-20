"""
=============================================================================
Caribbean AI Academy - Sixth Form (Ages 16-18)
Lesson 04: Reinforcement Learning
=============================================================================
Topic: Q-Learning Agent Navigating a Caribbean Island Grid

Bless up! Today we learning Reinforcement Learning (RL) — where an AI
agent learns by DOING, not by being told. Like how a young footballer
in Arima or Montego Bay learns: try something, see the result, adjust.

Our agent navigates an 8x8 Caribbean island grid:
- Start at a village on one side of the island
- Must reach the port on the other side to catch a boat
- Collect resources along the way (coconuts, fish, sugar cane)
- AVOID hurricanes and dangerous areas!

The agent learns through trial and error — just like life in the
Caribbean, yuh learn from experience!

Requirements: numpy
Author: Adrian Dunkley | Caribbean AI Academy
=============================================================================
"""

# 🧩 Puzzle Piece 7/12: Now put it all together... but you still need more pieces

import numpy as np
import random
import time

print("=" * 65)
print("  LESSON 04: Reinforcement Learning")
print("  Q-Learning on a Caribbean Island Grid")
print("=" * 65)


# =====================================================================
# SECTION 1: Understanding Reinforcement Learning
# =====================================================================
print("\n--- Part 1: What is Reinforcement Learning? ---\n")
print("""
Reinforcement Learning (RL) is different from other ML:

SUPERVISED LEARNING:  Teacher gives answers (like school exams)
UNSUPERVISED LEARNING: Find patterns yourself (like exploring a reef)
REINFORCEMENT LEARNING: Learn by doing + getting rewards/penalties
                        (like learning to fish — trial and error!)

Key Concepts:
  AGENT:       The learner/decision-maker (our island navigator)
  ENVIRONMENT: The world the agent lives in (the 8x8 island grid)
  STATE:       Where the agent is right now (grid position)
  ACTION:      What the agent can do (move up/down/left/right)
  REWARD:      Feedback signal (+10 for port, -50 for hurricane)
  POLICY:      The agent's strategy (which action to take in each state)

Think of it like a West Indies batsman at the crease:
  - STATE: score, overs remaining, wickets down, bowler type
  - ACTIONS: defend, attack, rotate strike, play for boundary
  - REWARD: runs scored (+), getting out (--), winning the match (+++)
  - POLICY: the strategy that emerges from experience
""")


# =====================================================================
# SECTION 2: Define the Caribbean Island Environment
# =====================================================================
print("--- Part 2: The Caribbean Island Grid Environment ---\n")

# Grid legend
EMPTY = 0        # Open path (sand, trail)
VILLAGE = 1      # Starting village
PORT = 2         # Destination (the port — goal!)
HURRICANE = 3    # Hurricane zone — DANGER! Big penalty
COCONUT = 4      # Coconut trees — small reward
FISH = 5         # Fishing spot — medium reward
SUGARCANE = 6    # Sugar cane field — medium reward
MOUNTAIN = 7     # Mountain — impassable (like Blue Mountains)
MANGROVE = 8     # Mangrove swamp — slow but passable, small penalty

CELL_NAMES = {
    0: '.', 1: 'V', 2: 'P', 3: 'H', 4: 'C',
    5: 'F', 6: 'S', 7: 'M', 8: '~'
}
CELL_DESCRIPTIONS = {
    0: 'Open Path', 1: 'Village (Start)', 2: 'Port (Goal)',
    3: 'Hurricane Zone', 4: 'Coconut Trees', 5: 'Fishing Spot',
    6: 'Sugar Cane', 7: 'Mountain', 8: 'Mangrove Swamp'
}

# Actions: Up, Down, Left, Right
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_DELTAS = {
    0: (-1, 0),   # UP
    1: (1, 0),    # DOWN
    2: (0, -1),   # LEFT
    3: (0, 1)     # RIGHT
}


class CaribbeanIslandEnv:
    """
    An 8x8 grid representing a Caribbean island.

    The agent starts at the Village (V) and must navigate to the
    Port (P) while collecting resources and avoiding hurricanes.

    Modelled after real Caribbean geography:
    - Mountains in the interior (like Jamaica's Blue Mountains)
    - Mangroves on the coast (like Trinidad's Caroni Swamp)
    - Hurricane zones (like the annual Atlantic hurricane belt)
    - Resources scattered across the island
    """

    def __init__(self):
        # Build the 8x8 island grid
        self.grid_size = 8
        self.grid = np.zeros((8, 8), dtype=int)

        # Place terrain features
        # Village (start) — bottom-left, like a fishing village
        self.start_pos = (7, 0)
        self.grid[7, 0] = VILLAGE

        # Port (goal) — top-right, like Kingston Harbour or Port of Spain
        self.goal_pos = (0, 7)
        self.grid[0, 7] = PORT

        # Hurricane zones — scattered danger areas
        hurricanes = [(1, 2), (2, 5), (4, 3), (5, 6), (6, 4)]
        for r, c in hurricanes:
            self.grid[r, c] = HURRICANE

        # Mountains — impassable interior (like Blue Mountains)
        mountains = [(3, 3), (3, 4), (4, 4)]
        for r, c in mountains:
            self.grid[r, c] = MOUNTAIN

        # Resources — scattered across the island
        self.grid[1, 1] = COCONUT    # Coconut grove
        self.grid[2, 3] = FISH       # Fishing village (like Oistins, Barbados)
        self.grid[5, 1] = SUGARCANE  # Sugar cane (like Trelawny, Jamaica)
        self.grid[6, 6] = COCONUT    # Another coconut area
        self.grid[3, 6] = FISH       # Fishing area
        self.grid[0, 3] = SUGARCANE  # Northern sugar fields

        # Mangrove swamps — passable but slow
        mangroves = [(7, 3), (6, 2), (1, 5)]
        for r, c in mangroves:
            self.grid[r, c] = MANGROVE

        self.agent_pos = self.start_pos
        self.collected_resources = []
        self.steps = 0
        self.max_steps = 100
        self.done = False

    def reset(self):
        """Reset the environment — new day on the island!"""
        self.agent_pos = self.start_pos
        self.collected_resources = []
        self.steps = 0
        self.done = False
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos):
        """Convert (row, col) position to a single state number."""
        return pos[0] * self.grid_size + pos[1]

    def _state_to_pos(self, state):
        """Convert state number back to (row, col)."""
        return (state // self.grid_size, state % self.grid_size)

    def step(self, action):
        """
        Take an action and return (new_state, reward, done, info).
        This is the core RL interface — agent acts, environment responds!
        """
        if self.done:
            return self._pos_to_state(self.agent_pos), 0, True, {}

        self.steps += 1
        dr, dc = ACTION_DELTAS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # Check boundaries — can't walk into the sea!
        if new_r < 0 or new_r >= self.grid_size or \
           new_c < 0 or new_c >= self.grid_size:
            return self._pos_to_state(self.agent_pos), -1, False, \
                   {'info': 'Hit island boundary!'}

        # Check for mountains — can't climb those!
        if self.grid[new_r, new_c] == MOUNTAIN:
            return self._pos_to_state(self.agent_pos), -1, False, \
                   {'info': 'Mountain blocking path!'}

        # Move the agent
        self.agent_pos = (new_r, new_c)
        cell = self.grid[new_r, new_c]

        # Calculate reward based on what's at the new position
        reward = -0.5  # Small penalty for each step (encourages efficiency)
        info = {}

        if cell == PORT:
            reward = 100  # Reached the port! Big reward!
            bonus = len(self.collected_resources) * 5
            reward += bonus
            self.done = True
            info['info'] = f'REACHED PORT! Resources collected: {len(self.collected_resources)}'

        elif cell == HURRICANE:
            reward = -50  # Hurricane damage — devastating!
            self.done = True
            info['info'] = 'HIT BY HURRICANE! Game over!'

        elif cell == COCONUT:
            reward = 5
            if (new_r, new_c) not in self.collected_resources:
                self.collected_resources.append((new_r, new_c))
                info['info'] = 'Collected coconuts!'

        elif cell == FISH:
            reward = 8
            if (new_r, new_c) not in self.collected_resources:
                self.collected_resources.append((new_r, new_c))
                info['info'] = 'Caught fish!'

        elif cell == SUGARCANE:
            reward = 6
            if (new_r, new_c) not in self.collected_resources:
                self.collected_resources.append((new_r, new_c))
                info['info'] = 'Harvested sugar cane!'

        elif cell == MANGROVE:
            reward = -3  # Mangroves slow yuh down
            info['info'] = 'Trudging through mangrove swamp...'

        # Time limit check
        if self.steps >= self.max_steps:
            self.done = True
            info['info'] = 'Ran out of time!'
            reward -= 20

        return self._pos_to_state(self.agent_pos), reward, self.done, info

    def render(self, show_agent=True):
        """Display the grid — visualize the island!"""
        print(f"\n  Island Grid (Step {self.steps}):")
        print("  " + "  ".join([str(c) for c in range(self.grid_size)]))
        print("  " + "--" * self.grid_size)
        for r in range(self.grid_size):
            row_str = f"{r}|"
            for c in range(self.grid_size):
                if show_agent and (r, c) == self.agent_pos:
                    row_str += " A"  # Agent position
                else:
                    row_str += f" {CELL_NAMES[self.grid[r, c]]}"
            print(f"  {row_str}")
        print(f"\n  Legend: V=Village A=Agent P=Port H=Hurricane")
        print(f"  C=Coconut F=Fish S=SugarCane M=Mountain ~=Mangrove")
        print(f"  Resources collected: {len(self.collected_resources)}")


# Create and display the environment
env = CaribbeanIslandEnv()
env.render(show_agent=True)


# =====================================================================
# SECTION 3: Q-Learning Algorithm
# =====================================================================
print("\n--- Part 3: Q-Learning Algorithm ---\n")
print("""
Q-LEARNING is one of the simplest RL algorithms. It builds a
"Q-table" — a cheat sheet that tells the agent:
  "In state S, if yuh take action A, yuh expected reward is Q(S,A)"

Like a Caribbean fisherman's mental map:
  "At Maracas Bay, if I cast my net east, I usually catch 10 fish"
  "At Oistins, if I go south at dawn, flying fish aplenty!"

The Q-value update formula:
  Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))

Where:
  alpha = learning rate (how fast yuh learn from new experience)
  gamma = discount factor (how much yuh value future vs immediate reward)
  s' = next state after taking action a in state s
""")


class QLearningAgent:
    """
    Q-Learning agent for navigating the Caribbean island.

    Think of the Q-table as the agent's "experience journal":
    - Each page = a grid position (state)
    - Each entry = expected reward for each action at that position
    - The agent fills this journal through exploration!
    """

    def __init__(self, num_states, num_actions, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate           # alpha
        self.gamma = discount_factor      # gamma
        self.epsilon = epsilon            # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        # Rows = states (64 grid positions), Cols = actions (4 directions)
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.

        Like a cricketer deciding:
        - EXPLORE (epsilon chance): Try something new — a reverse sweep?
        - EXPLOIT (1-epsilon chance): Play the shot yuh know works
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # Random action
        else:
            return np.argmax(self.q_table[state])  # Best known action

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-value using the Q-learning formula.
        This is where the magic happens — learning from experience!
        """
        # Current Q-value
        current_q = self.q_table[state, action]

        # Maximum Q-value for the next state (best possible future)
        if done:
            target = reward  # No future if episode is over
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # Update Q-value: blend old knowledge with new experience
        self.q_table[state, action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        """Reduce exploration over time — agent gets more confident."""
        self.epsilon = max(self.epsilon_min,
                          self.epsilon * self.epsilon_decay)


# =====================================================================
# SECTION 4: Training the Agent
# =====================================================================
print("--- Part 4: Training the Q-Learning Agent ---\n")

num_states = 64  # 8x8 grid
num_actions = 4  # Up, Down, Left, Right

agent = QLearningAgent(
    num_states=num_states,
    num_actions=num_actions,
    learning_rate=0.15,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.998,
    epsilon_min=0.01
)

# Training parameters
num_episodes = 2000
print(f"Training for {num_episodes} episodes...")
print(f"(Each episode = one attempt to reach the port)\n")

# Track training progress
rewards_history = []
successes = 0
hurricane_hits = 0

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    rewards_history.append(total_reward)
    agent.decay_epsilon()

    # Track outcomes
    if 'REACHED PORT' in info.get('info', ''):
        successes += 1
    elif 'HURRICANE' in info.get('info', ''):
        hurricane_hits += 1

    # Print progress every 400 episodes
    if (episode + 1) % 400 == 0:
        recent_avg = np.mean(rewards_history[-100:])
        print(f"  Episode {episode+1}/{num_episodes} | "
              f"Avg Reward (last 100): {recent_avg:.1f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Successes: {successes}")

print(f"\nTraining Complete!")
print(f"Total successes (reached port): {successes}/{num_episodes}")
print(f"Hurricane hits: {hurricane_hits}/{num_episodes}")
print(f"Final exploration rate: {agent.epsilon:.4f}")


# =====================================================================
# SECTION 5: Evaluate the Trained Agent
# =====================================================================
print("\n--- Part 5: Watching the Trained Agent ---\n")

# Run a few episodes with NO exploration (pure exploitation)
print("Running 10 test episodes (no exploration)...\n")
test_successes = 0
test_rewards = []

for test_ep in range(10):
    state = env.reset()
    total_reward = 0
    done = False
    path = [env.agent_pos]

    while not done:
        action = np.argmax(agent.q_table[state])  # Always best action
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        path.append(env.agent_pos)

    test_rewards.append(total_reward)
    outcome = info.get('info', 'Unknown')
    if 'REACHED PORT' in outcome:
        test_successes += 1

    print(f"  Test {test_ep+1}: Reward={total_reward:.1f} | "
          f"Steps={len(path)-1} | {outcome}")

print(f"\nTest Success Rate: {test_successes}/10")
print(f"Average Test Reward: {np.mean(test_rewards):.1f}")


# =====================================================================
# SECTION 6: Visualize the Learned Policy
# =====================================================================
print("\n--- Part 6: The Learned Policy (Best Action per Cell) ---\n")

ARROW = {0: '^', 1: 'v', 2: '<', 3: '>'}

print("  Best action at each grid position:")
print("  " + "  ".join([str(c) for c in range(8)]))
print("  " + "--" * 8)
for r in range(8):
    row_str = f"{r}|"
    for c in range(8):
        state = r * 8 + c
        cell = env.grid[r, c]
        if cell == MOUNTAIN:
            row_str += " M"
        elif cell == HURRICANE:
            row_str += " H"
        elif cell == PORT:
            row_str += " P"
        elif cell == VILLAGE:
            row_str += " V"
        else:
            best_action = np.argmax(agent.q_table[state])
            row_str += f" {ARROW[best_action]}"
    print(f"  {row_str}")

print("\n  Arrows show the best direction the agent learned!")
print("  Notice how it routes AROUND hurricanes and mountains.")


# =====================================================================
# SECTION 7: Visualize the Q-Table Heatmap (Text-based)
# =====================================================================
print("\n--- Part 7: Q-Value Heatmap (Max Q per State) ---\n")

print("  Brighter = higher expected reward from that position:")
levels = " .:-=+*#@"

max_q = np.max(agent.q_table, axis=1).reshape(8, 8)
q_min, q_max = max_q.min(), max_q.max()
q_range = q_max - q_min if q_max != q_min else 1

print("  " + "  ".join([str(c) for c in range(8)]))
print("  " + "--" * 8)
for r in range(8):
    row_str = f"{r}|"
    for c in range(8):
        val = (max_q[r, c] - q_min) / q_range
        idx = int(val * (len(levels) - 1))
        row_str += f" {levels[idx]}"
    row_str += f"  | max_Q = {max_q[r].max():.1f}"
    print(f"  {row_str}")


# =====================================================================
# SECTION 8: Caribbean Applications
# =====================================================================
print("\n--- Part 8: Caribbean RL Applications ---\n")
print("""
Reinforcement Learning is used (and could be used) across the Caribbean:

1. HURRICANE EVACUATION ROUTING (All Caribbean)
   - RL agents learn optimal evacuation routes in real-time
   - Like our grid agent but with real road networks
   - Consider traffic, shelter capacity, storm trajectory

2. ENERGY GRID MANAGEMENT (Trinidad, Jamaica, Barbados)
   - RL optimizes when to use solar/wind vs. fossil fuels
   - Agent learns to balance cost, reliability, and green energy
   - Like T&T transitioning from oil/gas to renewables

3. FISHERIES MANAGEMENT (Belize, Bahamas, Grenada)
   - RL decides fishing quotas and protected zones
   - Maximise sustainable catch — like our agent collecting resources
     without getting "hurricaned" (overfishing)

4. TRAFFIC OPTIMIZATION (Kingston, Port of Spain, Bridgetown)
   - RL controls traffic lights to minimize congestion
   - Agent learns rush hour patterns and adapts in real-time

5. CRICKET STRATEGY (West Indies!)
   - RL can learn optimal batting/bowling strategies
   - State: match situation, Action: shot/ball selection
   - Reward: runs scored or wickets taken
   - Like training a virtual Viv Richards or Curtly Ambrose!

6. TOURISM PRICING (All Caribbean)
   - RL learns dynamic pricing for hotels and flights
   - Maximise revenue across seasons while staying competitive
""")


# =====================================================================
# QUIZ
# =====================================================================
print("=" * 65)
print("  QUIZ: Reinforcement Learning")
print("=" * 65)
print("""
Q1: What are the five key components of an RL system?
    a) Data, Model, Loss, Optimizer, Accuracy
    b) Agent, Environment, State, Action, Reward
    c) Input, Hidden, Output, Weight, Bias
    d) Train, Test, Validate, Deploy, Monitor

Q2: What does the Q-table represent?
    a) A table of questions
    b) A lookup table storing expected cumulative rewards for each
       state-action pair
    c) A database of Caribbean quiz answers
    d) A multiplication table

Q3: What is epsilon-greedy exploration?
    a) Always taking the best known action
    b) With probability epsilon, take a random action (explore);
       otherwise take the best known action (exploit)
    c) Always taking random actions
    d) Taking the worst action on purpose

Q4: Why does epsilon decay over time?
    a) To save battery
    b) Early on, the agent needs to explore to discover good strategies;
       later, it should exploit what it learned
    c) Because epsilon always decays in nature
    d) To make the game harder

Q5: In our island grid, why does the agent receive -0.5 reward
    for each step?
    a) Because the island is unfriendly
    b) To encourage the agent to find the SHORTEST path to the port
       rather than wandering around forever
    c) Because negative rewards are more realistic
    d) To make training faster

Q6: What is the discount factor (gamma) and why is it important?
    a) A store discount code
    b) It determines how much the agent values future rewards vs.
       immediate rewards; gamma=0.95 means future rewards are
       almost as important as immediate ones
    c) It discounts bad actions
    d) It reduces the learning rate

Q7: How is the Q-learning update rule similar to how humans learn?
    a) It's not similar at all
    b) Like humans, it updates beliefs based on new experience —
       blending what yuh already know with what yuh just observed
    c) It requires sleep
    d) It needs a teacher

Q8: In our grid, why is the hurricane penalty (-50) much larger
    than the step penalty (-0.5)?
    a) Because hurricanes are expensive
    b) The magnitude of rewards/penalties shapes the agent's priorities;
       a large negative reward strongly discourages entering hurricane
       zones, modelling real catastrophic risk
    c) Because -50 is a round number
    d) To crash the program

Q9: If we changed the grid to 16x16, what would happen to training?
    a) Nothing would change
    b) The state space would increase from 64 to 256, requiring more
       episodes to explore and learn — but the algorithm stays the same
    c) The agent would get lost permanently
    d) Q-learning cannot handle larger grids

Q10: Design a Caribbean RL scenario (open-ended):
     Think of a real Caribbean problem where an agent could learn
     through trial and error. Define the state, actions, and rewards.
     (Discuss with your class!)

(Answers in quiz_answers.md)
""")

print("=" * 65)
print("  Lesson 04 Complete! Yuh trained an RL agent — big tings!")
print("  Next up: Generative Models (GANs)")
print("=" * 65)
