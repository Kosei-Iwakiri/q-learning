import random

NUM_STATES = 5
GOAL = NUM_STATES - 1
LEFT, RIGHT = 0, 1
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 1000
MAX_STEPS = 100

def step(state, action):
    """Move one cell and return (next_state, reward, finished)."""
    movement = -1 if action == LEFT else 1
    next_state = max(0, min(GOAL, state + movement))
    finished = next_state == GOAL
    reward = 10 if finished else -1
    return next_state, reward, finished


def best_action(values, rng):
    best_value = max(values)
    choices = [action for action, value in enumerate(values) if value == best_value]
    return rng.choice(choices)


def train():
    rng = random.Random(42)
    q_table = [[0.0, 0.0] for _ in range(NUM_STATES)]

    for episode in range(EPISODES):
        state = 0
        epsilon = max(0.05, 1.0 * (0.995 ** episode))

        for _ in range(MAX_STEPS):
            if rng.random() < epsilon:
                action = rng.choice([LEFT, RIGHT])
            else:
                action = best_action(q_table[state], rng)

            next_state, reward, finished = step(state, action)

            future_value = 0.0 if finished else max(q_table[next_state])
            target = reward + DISCOUNT * future_value
            old_value = q_table[state][action]
            q_table[state][action] += LEARNING_RATE * (target - old_value)

            state = next_state

            if finished:
                break

    return q_table

def show_world(state):
    cells = ["G" if position == GOAL else " " for position in range(NUM_STATES)]
    cells[state] = "A"
    print("".join(f"[{cell}]" for cell in cells))

def main():
    q_table = train()

    print("Learned Q-table (larger values are better):")
    print("State     Left    Right")

    for state, values in enumerate(q_table):
        label = " (goal)" if state == GOAL else ""
        print(f"{state:5}{values[LEFT]:9.2f}{values[RIGHT]:9.2f}{label}")

    print("\nWatch the trained agent: A = agent, G = goal")
    state = 0
    total_reward = 0
    rng = random.Random(123)
    show_world(state)

    for move in range(1, MAX_STEPS + 1):
        action = best_action(q_table[state], rng)
        state, reward, finished = step(state, action)
        total_reward += reward
        show_world(state)

        if finished:
            print(f"Reached the goal in {move} moves! Total reward: {total_reward}")
            break
    else: 
        print("The agent did not reach the goal within the step limit.")


if __name__ == "__main__":
    main()
