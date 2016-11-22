# coding: utf-8
import gym
import itertools
import matplotlib
import numpy as np
import sys
import collections
import chainer
import chainer.functions as F
import chainer.links as L

if "../" not in sys.path:
    sys.path.append("../") 
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')
env = CliffWalkingEnv()


class PolicyEstimator:

    def __init__(self):
        self.linear = L.Linear(env.observation_space.n, env.action_space.n,
                initialW=chainer.initializers.Zero(),
                initial_bias=chainer.initializers.Zero())
        self.optimizer = chainer.optimizers.Adam(alpha=0.01)
        self.optimizer.setup(self.linear)

    def predict(self, state):
        state = np.identity(env.observation_space.n, dtype=np.float32)[state]
        action_probs = F.flatten(F.softmax(self.linear(F.expand_dims(state, 0))))
        return action_probs.data

    def update(self, state, target, action):
        action = np.array([action], dtype=np.int32)
        state = np.identity(env.observation_space.n, dtype=np.float32)[state]
        action_probs = F.softmax(self.linear(F.expand_dims(state, 0)))
        selected_prob = F.select_item(action_probs, action)
        loss = -F.log(selected_prob) * target
        self.linear.cleargrads()
        loss.backward()
        self.optimizer.update()


class ValueEstimator:

    def __init__(self):
        self.linear = L.Linear(env.observation_space.n, 1,
                initialW=chainer.initializers.Zero(),
                initial_bias=chainer.initializers.Zero())
        self.optimizer = chainer.optimizers.Adam(alpha=0.1)
        self.optimizer.setup(self.linear)

    def predict(self, state):
        state = np.identity(env.observation_space.n, dtype=np.float32)[state]
        value = self.linear(F.expand_dims(state, 0))
        value = F.reshape(value, (1,))
        return value.data
    
    def update(self, state, target):
        state = np.identity(env.observation_space.n, dtype=np.float32)[state]
        value = self.linear(F.expand_dims(state, 0))
        value = F.reshape(value, (1,))
        loss = F.squared_difference(value, target)
        self.linear.cleargrads()
        loss.backward()
        self.optimizer.update()


def actor_critic(env, estimator_policy, estimator_value, num_episodes, 
        discount_factor=1.0):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    Transition = collections.namedtuple("Transition", 
         ["state", "action", "reward", "next_state", "done"])
    for i_episode in range(num_episodes):
        state = env.reset()
        episode = []
        for t in itertools.count():
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode.append(Transition(state=state, action=action, reward=reward, 
                next_state=next_state, done=done))
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)
            estimator_value.update(state, td_target)
            estimator_policy.update(state, td_error, action)
            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes, 
                stats.episode_rewards[i_episode - 1]), end="")
            if done: break
            state = next_state
    return stats


def reinforce(env, estimator_policy, estimator_value, num_episodes, 
        discount_factor=1.0):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    Transition = collections.namedtuple("Transition", 
        ["state", "action", "reward", "next_state", "done"])
    for i_episode in range(num_episodes):
        state = env.reset()
        episode = []
        for t in itertools.count():
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode.append(Transition(state=state, action=action, reward=reward, 
                next_state=next_state, done=done))
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, 
                    stats.episode_rewards[i_episode - 1]), end="")
            if done: break
            state = next_state
        for t, transition in enumerate(episode):
            total_return = sum(discount_factor**i * t.reward 
                    for i, t in enumerate(episode[t:]))
            total_return = np.array([total_return], dtype=np.float32)
            estimator_value.update(transition.state, total_return)
            baseline_value = estimator_value.predict(transition.state)            
            advantage = total_return - baseline_value
            estimator_policy.update(transition.state, advantage, transition.action)
    return stats


def debug(env, estimator_policy, estimator_value):
    state = env.reset()
    action_probs = estimator_policy.predict(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    next_state, reward, done, _ = env.step(action)
    value_next = estimator_value.predict(next_state)
    td_target = reward + value_next
    td_error = td_target - estimator_value.predict(state)
    estimator_value.update(state, td_target)
    estimator_policy.update(state, td_error, action)


estimator_policy = PolicyEstimator()
estimator_value = ValueEstimator()
# debug(env, estimator_policy, estimator_value)
# stats = reinforce(env, estimator_policy, estimator_value, 2000)
stats = actor_critic(env, estimator_policy, estimator_value, 300)
plotting.plot_episode_stats(stats, smoothing_window=10)
