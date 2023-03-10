import numpy as np
from helpers import tqdm_context


def rollout_sample(env, agent, replay_buffer, n_step, mode="train"):
    state = env.reset()
    rollout_return = 0

    for step in tqdm_context(range(n_step), desc="Episode", pos=1):
        # print(f'step {step}-------------')
        action, add_info = agent.get_action(state, time=env.t, mode=mode)
        next_state, reward, done = env.step(action)
        if mode == "train":
            replay_buffer.push(state, action, reward, next_state, done, add_info)

        rollout_return += reward
        state = next_state

    return rollout_return


def train_controller(agent, replay_buffer):
    train_it = agent.iterations
    if len(replay_buffer) > agent.batch_size:
        agent.train(replay_buffer, train_it)


# process episode rewards for multiple trials
def process_episode_rewards(many_episode_rewards):
    minimum = [np.min(episode_reward) for episode_reward in many_episode_rewards]
    maximum = [np.max(episode_reward) for episode_reward in many_episode_rewards]
    mean = [np.mean(episode_reward) for episode_reward in many_episode_rewards]

    return minimum, maximum, mean