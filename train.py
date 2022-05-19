from agent import DQNAgent

env = None
Agent = DQNAgent(state_size=4, action_size=1, seed=0)

num_episodes = 1000

for i in range(num_episodes):
    state = env.reset()
    score = 0
    while True:
        action = Agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        Agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    print("episode: {}/{}, score: {}".format(i, num_episodes, score))
