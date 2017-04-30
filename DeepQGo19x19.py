import gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from collections import deque
from random import sample
import matplotlib.pyplot as plt

env = gym.make('Go19x19-v0')
input_size = env.observation_space.sample().shape
output_size = env.action_space.n
hidden_size = input_size[0]*input_size[1]*input_size[2]

model = Sequential()
model.add(Flatten(input_shape=input_size))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(output_size, activation='linear'))
model.compile(optimizer=Adam(), loss='mean_squared_error')

#for keras
def resize(data):
    return data.reshape(1, data.shape[0], data.shape[1], data.shape[2])

def eGreedy(action, min_epsilon, step, decay, env):
    epsilon = min_epsilon + (1-min_epsilon)/(step*decay)
    if np.random.random() > epsilon:
        return action
    else:
        return env.action_space.sample()

def addMemory(memory, mem, mem_size):
    if len(memory) >= mem_size:
        memory.popleft()
        memory.append(mem)
    else:
        memory.append(mem)
    return

def actionReplay(memory, replay_length, reward_discount, model):
    if replay_length > len(memory):
        replay_length = len(memory)
    memory_sample = sample(memory, replay_length)
    x_train = []
    y_train = [model.predict(resize(i[0]))[0] for i in memory_sample]
    future_rewards = [model.predict(resize(i[3]))[0] if i[3] != 'T' else np.zeros(output_size) for i in memory_sample]
    for i, mem in enumerate(memory_sample):
        x_train.append(mem[0])
        target_reward = mem[2] + reward_discount * max(future_rewards[i])
        y_train[i][mem[1]] = target_reward
    model.fit(np.asarray(x_train), np.asarray(y_train), batch_size=replay_length, epochs=1, verbose=1)     

games = 10000
mem_size = 1000
memory = deque([])
replay_length = 64
reward_discount = 0.9
min_epsilon = 0.1
decay = 0.0001

y_gamelength = []
y_sumreward = []
x_step = []
step = 0
for game in range(0, games):
    obs = env.reset()
    game_length = 0
    sum_reward = 0
    while True:
        step += 1
        game_length += 1
        #env.render()
        observation = obs #store state for action replay
        action = np.argmax(model.predict(resize(obs))[0])
        action = eGreedy(action, min_epsilon, step, decay, env)
        obs, reward, done, info = env.step(action)
        if done:
            #Future observations all zeros for terminal state
            mem = (observation, action, reward, 'T')
            addMemory(memory, mem, mem_size)
            actionReplay(memory, replay_length, reward_discount, model)
            sum_reward += reward
            y_gamelength.append(game_length)
            if sum_reward == -1:
                sum_reward = 0
            else:
                sum_reward = 40
            y_sumreward.append(sum_reward)
            x_step.append(step)
            print('Game: ', game, 'Reward: ', reward)
            break
        mem = (observation, action, reward, obs)
        addMemory(memory, mem, mem_size)
        actionReplay(memory, replay_length, reward_discount, model)
        sum_reward += reward

plt.plot(np.asarray(x_step), np.asarray(y_gamelength), 'r-', np.asarray(x_step), np.asarray(y_sumreward), 'b-')
plt.xlabel('step')
plt.ylabel('game length / total reward')
plt.show()
