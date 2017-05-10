import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from random import sample
import copy

env = gym.make('Pendulum-v0')
#env = wrappers.Monitor(env, '/tmp/mountaincar-experiment-1', force=True)
input_size = env.observation_space.sample().shape[0]
output_size = env.action_space.sample().shape[0]

class Actor(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x)) 
        return x

class Critic(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.a = output_size
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50 + output_size, output_size)

    def forward(self, x, a):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(torch.cat((a,x)))) 
        return x
    
def addMem(memory, mem, memory_size):
    if len(memory) >= memory_size:
        memory.popleft()
    memory.append(mem)
    return

def eGreedy(action, e, step, decay, env):
    e = e + (1 - e) * np.exp(decay * step)
    if e > np.random.random():
        return env.action_space.sample()
    return action
    
def predictAct(actor, critic, x):
    x = torch.from_numpy(x).cuda()
    x = x.view(1, input_size)
    x = Variable(x.float())
    a = actor(x)
    return critic(x, a)

def predict(actor, critic, x):
    x = torch.from_numpy(x).cuda()
    x = Variable(x.float())
    a = actor(x)
    return critic(x, a)
    
def experienceReplay(memory, replay_length, net, target_net, discount, loss_func, step):
    if len(memory) < replay_length:
        batch_size = len(memory)
    else:
        batch_size = replay_length
    batch = sample(memory, batch_size)
    x_train = [mem[0] for mem in batch]
    q_future = [predictAct(target_actor, target_critic, mem[3]) if mem[3] != 'T' else
                Variable(torch.from_numpy(np.zeros(output_size)).cuda()) for mem in batch]
    q = predict(actor, critic, np.asarray(x_train))
    q_target = q.data.cpu().numpy()
    for i in range(len(q_target)):
        q_target[i, batch[i][1]] = batch[i][2] + discount * torch.max(q_future[i].data) 
    q_target = Variable(torch.from_numpy(q_target).cuda(), requires_grad=False)    
    loss = loss_func(q, q_target)
    loss.backward(retain_variables=True)
    optimizer.step()
    optimizer.zero_grad()
    if step % 1000 == 0:
        target_critic = copy.deepcopy(critic)
        target_actor = copy.deepcopy(actor)
    return loss.data.cpu().numpy()[0]

def initActor():
    net = Actor().cuda()
    net.zero_grad()
    return net

def initCritic():
    net = Critic().cuda()
    net.zero_grad()
    return net

actor = initActor()
target_actor = copy.deepcopy(actor)
critic = initCritic()
target_critic = copy.deepcopy(critic)
loss_func = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

iterations = 10000 
memory = deque([])
memory_size = 10000
replay_length = 100
discount = 0.9
epsilon = 0.1
decay = -.001
step = 0
sumLoss = 0
x = []
y = []
totalReward = 0

for it in range(iterations):
    obs = env.reset()
    while True:
        step += 1
        obs_prev = obs
        #env.render()
        action = np.argmax(predictAct(actor, critic, obs).data.cpu().numpy()[0])
        action = eGreedy(action, epsilon, step, decay, env)
        obs, reward, done, info = env.step(action)
        totalReward += reward
        if done:
            mem = (obs_prev, action, reward, 'T')
            addMem(memory, mem, memory_size)
            loss = experienceReplay(memory, replay_length, net, target_net, discount, loss_func, step)
            sumLoss += loss
            x.append(it)
            y.append(totalReward)
            totalReward = 0
            print 'loss on iteration', it, ': ', sumLoss/(100*memory_size)
            sumLoss = 0
            break
        mem = (obs_prev, action, reward, obs)
        addMem(memory, mem, memory_size)
        loss = experienceReplay(memory, replay_length, net, target_net, discount, loss_func, step)
        sumLoss += loss
plt.plot(x,y)            
plt.xlabel('iteration')
plt.ylabel('average error')
plt.grid(True)            
plt.show()
env.close()
