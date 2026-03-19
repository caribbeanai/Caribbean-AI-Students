"""
Reinforcement Learning Examples - Caribbean AI Curriculum
Designed by Adrian Dunkley (Adriandunkley.net) | FREE

Examples:
1. Caribbean Grid World Navigation (custom environment)
2. Caribbean Taxi Problem (custom environment)
3. Caribbean Island Resource Management Game (custom environment)

All examples are self-contained and runnable without gymnasium
(pure Python implementations for maximum accessibility).

Requirements:
    pip install numpy
    Optional: pip install gymnasium (for extended examples)
"""

import numpy as np
from collections import defaultdict
import random

# ============================================================================
# EXAMPLE 1: Caribbean Grid World Navigation
# ============================================================================

class CaribbeanGridWorld:
    """
    A simple grid world representing Caribbean island navigation.

    The agent (a sailor) must navigate from a starting port to a
    destination port while avoiding reefs and collecting treasures.

    Grid legend:
        S = Start (home port)
        G = Goal (destination port)
        R = Reef (danger! negative reward)
        T = Treasure (bonus reward)
        . = Open water (small negative reward to encourage efficiency)

    Caribbean Context:
        Think of this as navigating between Caribbean ports — from
        Kingston Harbour to Montego Bay, avoiding the reefs along
        the south coast of Jamaica.
    """

    def __init__(self):
        # 6x6 grid representing Caribbean waters
        self.grid = [
            ['S', '.', '.', 'R', '.', '.'],
            ['.', 'R', '.', '.', '.', 'R'],
            ['.', '.', 'T', '.', 'R', '.'],
            ['R', '.', '.', '.', '.', '.'],
            ['.', '.', 'R', '.', 'T', '.'],
            ['.', '.', '.', '.', '.', 'G'],
        ]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start = (0, 0)
        self.goal = (5, 5)
        self.state = self.start

        # Actions: 0=North, 1=East, 2=South, 3=West
        self.actions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.action_names = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
        self.n_actions = 4

        # Rewards
        self.rewards = {
            '.': -1,    # Open water — small cost to encourage efficiency
            'S': -1,    # Start
            'G': 100,   # Destination reached!
            'R': -20,   # Hit a reef!
            'T': 10,    # Found treasure!
        }

    def reset(self):
        """Reset to starting port."""
        self.state = self.start
        return self.state

    def step(self, action):
        """Take an action, return (new_state, reward, done)."""
        dr, dc = self.actions[action]
        new_r = max(0, min(self.rows - 1, self.state[0] + dr))
        new_c = max(0, min(self.cols - 1, self.state[1] + dc))
        self.state = (new_r, new_c)

        cell = self.grid[new_r][new_c]
        reward = self.rewards[cell]
        done = (self.state == self.goal)

        return self.state, reward, done

    def render(self, q_table=None):
        """Display the grid with agent position and optional policy arrows."""
        arrow_map = {0: '^', 1: '>', 2: 'v', 3: '<'}
        print("\n  Caribbean Waters Grid:")
        print("  " + "---" * self.cols)
        for r in range(self.rows):
            row_str = "  "
            for c in range(self.cols):
                if (r, c) == self.state:
                    row_str += " @ "  # Agent position
                elif q_table is not None and self.grid[r][c] not in ['G']:
                    state = (r, c)
                    if state in q_table and max(q_table[state]) != 0:
                        best_action = np.argmax(q_table[state])
                        row_str += f" {arrow_map[best_action]} "
                    else:
                        row_str += f" {self.grid[r][c]} "
                else:
                    row_str += f" {self.grid[r][c]} "
            print(row_str)
        print("  " + "---" * self.cols)
        print("  Legend: S=Start, G=Goal, R=Reef, T=Treasure, @=Agent")


def q_learning_caribbean_gridworld():
    """
    Train an agent to navigate Caribbean waters using Q-Learning.

    This is the fundamental RL algorithm — the foundation for
    understanding more complex methods.
    """
    print("=" * 70)
    print("EXAMPLE 1: Caribbean Grid World Navigation (Q-Learning)")
    print("=" * 70)
    print("\nA sailor must navigate from Kingston (S) to Montego Bay (G)")
    print("while avoiding reefs (R) and collecting treasure (T).")

    env = CaribbeanGridWorld()

    # Q-Learning hyperparameters
    alpha = 0.1          # Learning rate
    gamma = 0.95         # Discount factor
    epsilon = 1.0        # Exploration rate (start fully exploratory)
    epsilon_min = 0.01   # Minimum exploration
    epsilon_decay = 0.995  # Decay per episode
    n_episodes = 1000

    # Initialize Q-table as a dictionary of arrays
    q_table = defaultdict(lambda: np.zeros(env.n_actions))

    # Training metrics
    rewards_per_episode = []
    steps_per_episode = []

    print(f"\n--- Training Q-Learning Agent ---")
    print(f"  Episodes: {n_episodes}")
    print(f"  Learning rate: {alpha}")
    print(f"  Discount factor: {gamma}")
    print(f"  Epsilon decay: {epsilon} -> {epsilon_min}")

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 100:  # Max 100 steps per episode
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, env.n_actions - 1)  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            # Take action
            next_state, reward, done = env.step(action)

            # Q-Learning update
            best_next_q = np.max(q_table[next_state])
            td_target = reward + gamma * best_next_q * (1 - done)
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Log progress
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            avg_steps = np.mean(steps_per_episode[-100:])
            print(f"  Episode {episode+1:4d}: Avg Reward={avg_reward:.1f}, "
                  f"Avg Steps={avg_steps:.1f}, Epsilon={epsilon:.3f}")

    # Show results
    print(f"\n--- Training Complete ---")
    print(f"  Final 100-episode avg reward: {np.mean(rewards_per_episode[-100:]):.1f}")
    print(f"  Final 100-episode avg steps:  {np.mean(steps_per_episode[-100:]):.1f}")

    # Show learned policy
    print("\n--- Learned Navigation Policy ---")
    env.reset()
    env.render(q_table)

    # Run a demonstration episode
    print("\n--- Demonstration: Optimal Route ---")
    state = env.reset()
    path = [state]
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < 20:
        action = np.argmax(q_table[state])
        state, reward, done = env.step(action)
        path.append(state)
        total_reward += reward
        steps += 1
        cell = env.grid[state[0]][state[1]]
        action_name = env.action_names[action]
        print(f"  Step {steps}: {action_name} -> {state} "
              f"[{cell}] Reward: {reward:+d}")

    print(f"\n  Total reward: {total_reward}")
    print(f"  Path length: {len(path)} steps")
    print(f"  Reached destination: {'Yes!' if done else 'No (max steps reached)'}")

    return q_table


# ============================================================================
# EXAMPLE 2: Caribbean Taxi Problem
# ============================================================================

class CaribbeanTaxi:
    """
    A Caribbean taxi driver must pick up and drop off passengers
    at various locations in a Caribbean city.

    Locations:
        0: Airport (Norman Manley, Kingston)
        1: Downtown (Parade, Kingston)
        2: Uptown (Half Way Tree)
        3: Beach (Port Royal)
        4: Market (Coronation Market)

    The taxi operates on a 5x5 grid. Passengers appear at random
    locations with random destinations.

    Actions:
        0: North, 1: East, 2: South, 3: West
        4: Pick up passenger, 5: Drop off passenger
    """

    def __init__(self):
        self.grid_size = 5
        self.locations = {
            0: (0, 0),   # Airport
            1: (0, 4),   # Downtown
            2: (4, 0),   # Uptown
            3: (4, 4),   # Beach
            4: (2, 2),   # Market
        }
        self.location_names = {
            0: 'Airport', 1: 'Downtown', 2: 'Uptown',
            3: 'Beach', 4: 'Market'
        }
        self.n_actions = 6
        self.action_names = {
            0: 'North', 1: 'East', 2: 'South', 3: 'West',
            4: 'Pickup', 5: 'Dropoff'
        }
        self.reset()

    def reset(self):
        """Reset with random taxi position, passenger, and destination."""
        self.taxi_pos = (random.randint(0, 4), random.randint(0, 4))
        self.passenger_loc = random.randint(0, 4)  # Location index
        self.destination = random.choice(
            [i for i in range(5) if i != self.passenger_loc]
        )
        self.has_passenger = False
        return self._get_state()

    def _get_state(self):
        """Encode state as a tuple."""
        return (self.taxi_pos[0], self.taxi_pos[1],
                self.passenger_loc, self.destination,
                int(self.has_passenger))

    def step(self, action):
        """Execute an action."""
        reward = -1  # Default: small penalty per step (encourages efficiency)
        done = False

        if action == 0:  # North
            self.taxi_pos = (max(0, self.taxi_pos[0] - 1), self.taxi_pos[1])
        elif action == 1:  # East
            self.taxi_pos = (self.taxi_pos[0], min(4, self.taxi_pos[1] + 1))
        elif action == 2:  # South
            self.taxi_pos = (min(4, self.taxi_pos[0] + 1), self.taxi_pos[1])
        elif action == 3:  # West
            self.taxi_pos = (self.taxi_pos[0], max(0, self.taxi_pos[1] - 1))
        elif action == 4:  # Pickup
            passenger_pos = self.locations[self.passenger_loc]
            if self.taxi_pos == passenger_pos and not self.has_passenger:
                self.has_passenger = True
                reward = 5  # Successful pickup
            else:
                reward = -10  # Wrong pickup attempt
        elif action == 5:  # Dropoff
            dest_pos = self.locations[self.destination]
            if self.taxi_pos == dest_pos and self.has_passenger:
                reward = 50  # Successful delivery!
                done = True
            else:
                reward = -10  # Wrong dropoff attempt

        return self._get_state(), reward, done

    def render(self):
        """Display the current state."""
        print(f"\n  Taxi at: {self.taxi_pos}")
        print(f"  Passenger at: {self.location_names[self.passenger_loc]} "
              f"{self.locations[self.passenger_loc]}")
        print(f"  Destination: {self.location_names[self.destination]} "
              f"{self.locations[self.destination]}")
        print(f"  Has passenger: {self.has_passenger}")


def q_learning_caribbean_taxi():
    """
    Train a Caribbean taxi driver using Q-Learning.

    The driver learns to efficiently pick up and deliver passengers
    across Kingston, Jamaica.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Caribbean Taxi Problem (Q-Learning)")
    print("=" * 70)
    print("\nA Kingston taxi driver must learn to:")
    print("  1. Navigate to the passenger")
    print("  2. Pick them up")
    print("  3. Navigate to the destination")
    print("  4. Drop them off")
    print("\nLocations: Airport, Downtown, Uptown, Beach, Market")

    env = CaribbeanTaxi()

    # Q-Learning parameters
    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9995
    n_episodes = 5000

    # Q-table
    q_table = defaultdict(lambda: np.zeros(env.n_actions))

    rewards_per_episode = []

    print(f"\n--- Training Taxi Driver ---")
    print(f"  Episodes: {n_episodes}")

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            if random.random() < epsilon:
                action = random.randint(0, env.n_actions - 1)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)

            best_next_q = np.max(q_table[next_state])
            td_target = reward + gamma * best_next_q * (1 - done)
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-500:])
            print(f"  Episode {episode+1:5d}: Avg Reward={avg_reward:.1f}, "
                  f"Epsilon={epsilon:.4f}, Q-states={len(q_table)}")

    print(f"\n--- Training Complete ---")
    print(f"  States explored: {len(q_table)}")
    print(f"  Final 500-episode avg reward: {np.mean(rewards_per_episode[-500:]):.1f}")

    # Demonstration
    print("\n--- Demonstration: Trained Taxi Driver ---")
    state = env.reset()
    env.render()
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < 30:
        action = np.argmax(q_table[state])
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        action_name = env.action_names[action]
        print(f"  Step {steps}: {action_name:10s} -> Taxi at {env.taxi_pos}, "
              f"Has passenger: {env.has_passenger}, Reward: {reward:+d}")

    print(f"\n  Total reward: {total_reward}")
    print(f"  Delivery {'successful!' if done else 'failed (max steps reached)'}")

    # Compare trained vs random
    print("\n--- Comparison: Trained vs Random Driver ---")
    n_test = 100

    # Trained driver
    trained_rewards = []
    trained_steps = []
    for _ in range(n_test):
        state = env.reset()
        total_r = 0
        s = 0
        done = False
        while not done and s < 200:
            action = np.argmax(q_table[state])
            state, reward, done = env.step(action)
            total_r += reward
            s += 1
        trained_rewards.append(total_r)
        trained_steps.append(s)

    # Random driver
    random_rewards = []
    random_steps = []
    for _ in range(n_test):
        state = env.reset()
        total_r = 0
        s = 0
        done = False
        while not done and s < 200:
            action = random.randint(0, env.n_actions - 1)
            state, reward, done = env.step(action)
            total_r += reward
            s += 1
        random_rewards.append(total_r)
        random_steps.append(s)

    print(f"  {'Metric':<25s} {'Trained':>10s} {'Random':>10s}")
    print(f"  {'-'*47}")
    print(f"  {'Avg Reward':<25s} {np.mean(trained_rewards):>10.1f} {np.mean(random_rewards):>10.1f}")
    print(f"  {'Avg Steps':<25s} {np.mean(trained_steps):>10.1f} {np.mean(random_steps):>10.1f}")
    print(f"  {'Success Rate':<25s} {sum(1 for r in trained_rewards if r > 0)/n_test:>10.1%} "
          f"{sum(1 for r in random_rewards if r > 0)/n_test:>10.1%}")

    return q_table


# ============================================================================
# EXAMPLE 3: Caribbean Island Resource Management Game
# ============================================================================

class CaribbeanIslandManager:
    """
    Manage resources on a Caribbean island over 12 months (1 year).

    The agent is the island's resource manager and must balance:
    - Tourism development
    - Environmental conservation
    - Agriculture
    - Energy management

    State:
        - Month (1-12, affects tourism and weather)
        - Treasury balance (JMD millions)
        - Environment health (0-100)
        - Tourism satisfaction (0-100)
        - Food security (0-100)
        - Energy reserves (0-100)

    Actions:
        0: Invest in tourism infrastructure
        1: Invest in environmental conservation
        2: Invest in agriculture
        3: Invest in renewable energy
        4: Balance all sectors equally
        5: Save money (no investment)

    Caribbean Context:
        Small island developing states face unique resource management
        challenges. Climate change, tourism dependency, food import
        reliance, and energy costs are real concerns from Jamaica to
        Barbados to the Eastern Caribbean.
    """

    def __init__(self):
        self.n_actions = 6
        self.action_names = {
            0: 'Tourism', 1: 'Environment', 2: 'Agriculture',
            3: 'Energy', 4: 'Balance All', 5: 'Save Money'
        }
        self.reset()

    def reset(self):
        """Reset island to starting conditions."""
        self.month = 1
        self.treasury = 50.0     # JMD millions
        self.environment = 70.0  # Health score
        self.tourism_sat = 60.0  # Satisfaction score
        self.food_security = 60.0
        self.energy = 50.0
        return self._get_state()

    def _get_state(self):
        """Discretize state for Q-learning."""
        return (
            self.month,
            int(self.treasury / 10),
            int(self.environment / 20),
            int(self.tourism_sat / 20),
            int(self.food_security / 20),
            int(self.energy / 20)
        )

    def step(self, action):
        """Advance one month with the chosen investment."""
        # Seasonal effects
        is_tourist_season = self.month in [12, 1, 2, 3, 4]
        is_hurricane_season = self.month in [6, 7, 8, 9, 10, 11]
        is_planting_season = self.month in [3, 4, 5, 9, 10]

        # Base income from tourism (seasonal)
        tourism_income = 15 if is_tourist_season else 8
        tourism_income *= (self.tourism_sat / 100) * (self.environment / 100)

        # Agriculture income
        agri_income = 5 * (self.food_security / 100)

        # Energy costs
        energy_cost = 8 * (1 - self.energy / 200)  # Better energy = lower cost

        # Base treasury change
        self.treasury += tourism_income + agri_income - energy_cost

        # Apply action effects
        investment_cost = 10  # Base investment cost

        if action == 0:  # Tourism
            self.treasury -= investment_cost
            self.tourism_sat = min(100, self.tourism_sat + 8)
            self.environment -= 3  # Tourism can harm environment
        elif action == 1:  # Environment
            self.treasury -= investment_cost
            self.environment = min(100, self.environment + 10)
            self.tourism_sat += 2  # Healthy environment attracts tourists
        elif action == 2:  # Agriculture
            self.treasury -= investment_cost
            self.food_security = min(100, self.food_security + 10)
            if is_planting_season:
                self.food_security = min(100, self.food_security + 5)  # Bonus
        elif action == 3:  # Energy
            self.treasury -= investment_cost
            self.energy = min(100, self.energy + 12)
        elif action == 4:  # Balance
            self.treasury -= investment_cost
            self.tourism_sat = min(100, self.tourism_sat + 3)
            self.environment = min(100, self.environment + 3)
            self.food_security = min(100, self.food_security + 3)
            self.energy = min(100, self.energy + 3)
        elif action == 5:  # Save
            pass  # No investment, no improvement

        # Natural decay (things deteriorate without investment)
        self.environment -= 2
        self.tourism_sat -= 2
        self.food_security -= 3  # Food security degrades faster (imports needed)
        self.energy -= 2

        # Hurricane impact during hurricane season
        if is_hurricane_season and random.random() < 0.15:
            damage = random.uniform(5, 20)
            self.environment -= damage * 0.5
            self.tourism_sat -= damage
            self.treasury -= damage * 0.5
            # Better infrastructure mitigates damage slightly

        # Clip values
        self.environment = np.clip(self.environment, 0, 100)
        self.tourism_sat = np.clip(self.tourism_sat, 0, 100)
        self.food_security = np.clip(self.food_security, 0, 100)
        self.energy = np.clip(self.energy, 0, 100)
        self.treasury = max(-50, self.treasury)  # Can go into debt

        # Calculate reward
        # Reward is based on overall wellbeing
        wellbeing = (
            0.2 * self.treasury / 50 +           # Financial health
            0.25 * self.environment / 100 +        # Environmental health
            0.2 * self.tourism_sat / 100 +         # Tourism
            0.2 * self.food_security / 100 +       # Food security
            0.15 * self.energy / 100               # Energy
        )
        reward = wellbeing * 10

        # Penalties for critical failures
        if self.environment < 20:
            reward -= 5  # Environmental crisis
        if self.food_security < 20:
            reward -= 5  # Food crisis
        if self.treasury < 0:
            reward -= 3  # Debt

        # Advance month
        self.month = (self.month % 12) + 1
        done = (self.month == 1 and self.treasury > 0)  # Year complete

        # End early if bankrupt and all sectors failing
        if self.treasury < -40:
            done = True
            reward -= 20

        return self._get_state(), reward, done

    def render(self):
        """Display current island status."""
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print(f"\n  Month: {month_names[self.month]} | "
              f"Treasury: ${self.treasury:.1f}M JMD")
        print(f"  Environment: {self.environment:.0f}/100 | "
              f"Tourism: {self.tourism_sat:.0f}/100")
        print(f"  Food Security: {self.food_security:.0f}/100 | "
              f"Energy: {self.energy:.0f}/100")


def q_learning_island_management():
    """
    Train an RL agent to manage a Caribbean island's resources.

    This example shows how RL can be applied to complex
    multi-objective resource management problems faced by
    Caribbean small island developing states.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Caribbean Island Resource Management Game")
    print("=" * 70)
    print("\nManage a Caribbean island for one year!")
    print("Balance: Tourism, Environment, Agriculture, Energy, Treasury")
    print("Actions: Invest in Tourism/Environment/Agriculture/Energy,")
    print("         Balance All, or Save Money")

    env = CaribbeanIslandManager()

    # Q-Learning parameters
    alpha = 0.15
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9995
    n_episodes = 10000

    q_table = defaultdict(lambda: np.zeros(env.n_actions))
    rewards_per_episode = []

    print(f"\n--- Training Island Manager ---")
    print(f"  Episodes: {n_episodes}")

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 12:  # Max 12 months
            if random.random() < epsilon:
                action = random.randint(0, env.n_actions - 1)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)

            best_next_q = np.max(q_table[next_state])
            td_target = reward + gamma * best_next_q * (1 - done)
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % 2000 == 0:
            avg_reward = np.mean(rewards_per_episode[-500:])
            print(f"  Episode {episode+1:5d}: Avg Reward={avg_reward:.1f}, "
                  f"Epsilon={epsilon:.4f}")

    print(f"\n--- Training Complete ---")
    print(f"  States explored: {len(q_table)}")
    print(f"  Final avg reward: {np.mean(rewards_per_episode[-500:]):.1f}")

    # Demonstration: watch the trained agent manage the island for a year
    print("\n--- Demonstration: Trained Island Manager (1 Year) ---")
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    state = env.reset()
    print(f"\n  Starting conditions:")
    env.render()

    total_reward = 0
    decisions = []

    for month in range(12):
        action = np.argmax(q_table[state])
        state, reward, done = env.step(action)
        total_reward += reward
        decisions.append(env.action_names[action])
        current_month = month_names[((month) % 12) + 1]
        print(f"\n  {current_month}: Invested in {env.action_names[action]:15s} "
              f"| Reward: {reward:.1f}")
        env.render()
        if done:
            break

    print(f"\n  --- Year-End Summary ---")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Final treasury: ${env.treasury:.1f}M JMD")
    print(f"  Final environment: {env.environment:.0f}/100")
    print(f"  Final tourism: {env.tourism_sat:.0f}/100")
    print(f"  Final food security: {env.food_security:.0f}/100")
    print(f"  Final energy: {env.energy:.0f}/100")

    # Analyze strategy
    print(f"\n  Investment decisions: {decisions}")
    from collections import Counter
    decision_counts = Counter(decisions)
    print(f"  Strategy breakdown:")
    for action_name, count in decision_counts.most_common():
        bar = "=" * (count * 5)
        print(f"    {action_name:15s}: {count:2d} months {bar}")

    # Compare with random management
    print("\n--- Comparison: Trained vs Random Manager (100 years) ---")
    n_test = 100

    trained_rewards = []
    random_rewards = []

    for _ in range(n_test):
        # Trained
        state = env.reset()
        tr = 0
        for _ in range(12):
            action = np.argmax(q_table[state])
            state, r, d = env.step(action)
            tr += r
            if d:
                break
        trained_rewards.append(tr)

        # Random
        state = env.reset()
        rr = 0
        for _ in range(12):
            action = random.randint(0, env.n_actions - 1)
            state, r, d = env.step(action)
            rr += r
            if d:
                break
        random_rewards.append(rr)

    print(f"  {'Metric':<25s} {'Trained':>10s} {'Random':>10s}")
    print(f"  {'-'*47}")
    print(f"  {'Avg Annual Reward':<25s} {np.mean(trained_rewards):>10.1f} {np.mean(random_rewards):>10.1f}")
    print(f"  {'Best Year':<25s} {max(trained_rewards):>10.1f} {max(random_rewards):>10.1f}")
    print(f"  {'Worst Year':<25s} {min(trained_rewards):>10.1f} {min(random_rewards):>10.1f}")

    return q_table


# ============================================================================
# MAIN — Run All Examples
# ============================================================================

def main():
    """Run all Caribbean reinforcement learning examples."""
    print("+" * 70)
    print("+   Caribbean AI Curriculum — Reinforcement Learning Examples       +")
    print("+   Designed by Adrian Dunkley (Adriandunkley.net) | FREE           +")
    print("+" * 70)
    print()
    print("Three RL examples featuring Caribbean scenarios:")
    print("  1. Caribbean Grid World Navigation")
    print("  2. Caribbean Taxi Problem")
    print("  3. Caribbean Island Resource Management Game")
    print()

    q_learning_caribbean_gridworld()
    q_learning_caribbean_taxi()
    q_learning_island_management()

    print("\n" + "=" * 70)
    print("All reinforcement learning examples complete!")
    print("From navigating Caribbean waters to managing island resources,")
    print("RL teaches agents to make decisions through trial and error.")
    print("Try modifying the environments — change rewards, add complexity,")
    print("or create yuh own Caribbean RL challenges!")
    print("=" * 70)


if __name__ == "__main__":
    main()
