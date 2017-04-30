import gym
from gym import wrappers
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from gym.wrappers import Monitor
import matplotlib.pyplot as plt
from collections import deque
from random import sample, random

def eGreedy(it, action, env):
    e = 0.01 + (1 - 0.01) * np.exp(-0.001 * it)
    if e > random():
        return env.action_space.sample()
    return action

def AddMem(mem, data, maxsize):
    if len(mem) < maxsize:
        mem.append(data)
    else:
        mem.popleft()
        mem.append(data)  
    return

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
iterations = 200
#learning_rate = 0.3
mem_maxsize = 10000
batch_size = 100
discount = 0.9
input_size = env.observation_space.high.shape[0]
output_size = env.action_space.n
mem = deque([])

model = Sequential()

model.add(Dense(50, activation='relu', input_dim=input_size))

model.add(Dense(output_size, activation='linear'))

model.compile(loss='mse', optimizer='adam')

totalReward = 0
step = 0
x = []
y = []
print("Running...")
for i in range (1,iterations):
    observation = env.reset()
    while True:
        step += 1
        if i > 0:
            env.render()
        action = np.argmax(model.predict(observation.reshape(1,input_size))[0])
        action = eGreedy(step, action, env)
        obs = observation 
        observation, reward, done, info = env.step(action)
        if done: reward = -100
        AddMem(mem, tuple((obs,action,reward,observation)), mem_maxsize)
        totalReward += reward
        if len(mem) > 1:
            batch_s = min(batch_size, len(mem))
            batch = sample(mem, batch_s)
            x_train = np.zeros((batch_s,input_size))
            y_train = np.zeros((batch_s,output_size))
            state = np.array([s[0] for s in batch])
            next_state = np.array([s[3] for s in batch])
            q = model.predict(state)
            q_next = model.predict(next_state)
            for m in range(0,batch_s):
                obv,action,reward,next_obv = batch[m]
                if np.sum(obv) == 0:
                    continue
                target = q[m] 
                target[action] = reward + (discount * np.amax(q_next[m]))
                x_train[m] = state[m]
                y_train[m] = target
            model.fit(x_train, y_train, batch_size=batch_s, epochs=1, verbose=0) 
        if done:
            z_obv = np.zeros(input_size)
            AddMem(mem, tuple((z_obv,0,0,z_obv)), mem_maxsize)
            x.append(i)
            y.append(totalReward)
            totalReward = 0
            break
plt.plot(x,y)            
plt.xlabel('iteration')
plt.ylabel('average error')
plt.grid(True)            
plt.show()
env.close()
print("Done")
