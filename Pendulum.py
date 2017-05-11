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
import sys

env = gym.make('Pendulum-v0')
env = wrappers.Monitor(env, '/tmp/pendulum-experiment-1', force=True)
input_size = env.observation_space.sample().shape[0]
output_size = env.action_space.sample().shape[0]
has_cuda = torch.cuda.is_available()

class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.hardtanh(self.fc3(x))
        return x

class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.a = output_size
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50 + output_size, 1)

    def forward(self, x, a):
        x = F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x = self.fc3(torch.cat((a,x),1))
        return x
    
def addMem(memory, mem, memory_size):
    if len(memory) >= memory_size:
        memory.popleft()
    memory.append(mem)
    return

def eGreedy(action, e, step, decay, env):
    e = e + (1 - e) * np.exp(decay * step)
    if e > np.random.random():
        if has_cuda:
            action.data = torch.from_numpy(env.action_space.sample()).float().cuda()
        else:    
            action.data = torch.from_numpy(env.action_space.sample()).float()
    return 
    
def predictAct(actor, x):
    x = torch.from_numpy(x).float()
    if has_cuda:
        x = x.cuda()
    x = x.view(1, input_size)
    x = Variable(x)
    return actor(x)

def predictActs(actor, x):
    x = torch.from_numpy(x).float()
    if has_cuda:
        x = x.cuda()
    x = Variable(x)
    return actor(x)

def critiqueOne(critic, x, a):
    x = np.atleast_2d(x)
    #a = a.unsqueeze(0).float()
    x = torch.from_numpy(x).float()
    if has_cuda:
        x = x.cuda()
        a = a.cuda()
    x = x.view(1, input_size)
    x = Variable(x)
    return critic(x, a)

def critique(critic, x, a):
    x = np.atleast_2d(x)
    x = torch.from_numpy(x).float()
    if has_cuda:
        x = x.cuda()
        a = a.cuda()
    x = Variable(x)
    return critic(x, a)
    
def experienceReplay(memory, replay_length, actor, target_actor, critic,
                     target_critic, discount, loss_func, step):
    if len(memory) < replay_length:
        return 0, 0
    else:
        batch_size = replay_length
    batch = sample(memory, batch_size)
    x_train = np.asarray([mem[0] for mem in batch])
    acts = [mem[1].float() for mem in batch]
    acts = torch.stack(acts)
    zero = Variable(torch.from_numpy(np.zeros(output_size)).float())
    q_future = [critiqueOne(target_critic, mem[3], predictAct(target_actor, mem[3]))
                if mem[3] != 'T' else zero for mem in batch]
    q = critique(critic, x_train, acts)
    q_target = q.data.cpu().clone().numpy()
    for i in range(len(q_target)):
        q_target[i] = (batch[i][2] + discount * q_future[i].data.cpu().numpy()[0])
    if has_cuda:
        q_target = Variable(torch.from_numpy(q_target).cuda(), volatile=False)
    else:
        q_target = Variable(torch.from_numpy(q_target), volatile=False) 
    critic_optimizer.zero_grad()    
    loss = loss_func(q, q_target)
    loss.backward(retain_variables=True)   
    critic_optimizer.step()
    actor_optimizer.zero_grad()
    target_acts = predictActs(actor, x_train)
    p_loss = critique(target_critic, x_train, target_acts)
    p_loss = p_loss.mean()
    p_loss.backward(retain_variables=True)
    actor_optimizer.step()
    if step % 2000 == 0:
        print('---------------Update Target Networks----------------')
        target_critic = copy.deepcopy(critic)
        target_actor = copy.deepcopy(actor)
    return loss.data.cpu().numpy()[0], p_loss.data.cpu().numpy()[0]

def initActor():
    net = Actor()
    if has_cuda:
        net = net.cuda()
    net.zero_grad()
    return net

def initCritic():
    net = Critic()
    if has_cuda:
        net = net.cuda()
    net.zero_grad()
    return net

actor = initActor()
target_actor = copy.deepcopy(actor)
critic = initCritic()
target_critic = copy.deepcopy(critic)
loss_func = nn.MSELoss()
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

iterations = 500
memory = deque([])
memory_size = 10000
replay_length = 100
discount = 0.9
epsilon = 0.1
decay = -.01
step = 0
sumLoss = 0
p_sumLoss = 0
x = []
y = []
totalReward = 0

for it in range(iterations):
    obs = env.reset().reshape(3)
    curr_step = step
    while True:
        step += 1
        obs_prev = obs
        #env.render()
        action = predictAct(actor, obs)[0]
        eGreedy(action, epsilon, step, decay, env)
        obs, reward, done, info = env.step(action.data.cpu().numpy())
        obs = obs.reshape(3)
        totalReward += reward
        if done:
            mem = (obs_prev, action, reward, 'T')
            addMem(memory, mem, memory_size)
            loss, p_loss = experienceReplay(memory, replay_length, actor, target_actor,
                                            critic, target_critic, discount, loss_func, step)
            sumLoss += loss
            p_sumLoss += p_loss
            x.append(it)
            y.append(totalReward)
            print 'reward on iteration', it, ': ', totalReward
            totalReward = 0
            print 'average loss on iteration', it, ': ', sumLoss/(step-curr_step)
            sumLoss = 0
            print 'average p_loss on iteration', it, ': ', p_sumLoss/(step-curr_step)
            p_sumLoss = 0
            print '--------------------------------------------------------'
            break
        mem = (obs_prev, action, reward, obs)
        addMem(memory, mem, memory_size)
        loss, p_loss = experienceReplay(memory, replay_length, actor, target_actor,
                                        critic, target_critic, discount, loss_func, step)
        sumLoss += loss
        p_sumLoss += p_loss
plt.plot(x,y)            
plt.xlabel('iteration')
plt.ylabel('reward')
plt.grid(True)            
plt.show()
