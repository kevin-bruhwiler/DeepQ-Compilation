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
from scipy.misc import imresize
import copy

env = gym.make('Breakout-v0')
env = wrappers.Monitor(env, '/tmp/breakout-experiment-1', force=True)
input_size = env.observation_space.sample().shape[0]
output_size = env.action_space.n

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(84, 32, 4)
        self.conv2 = nn.Conv1d(32, 64, 4)
        self.conv3 = nn.Conv1d(64, 64, 4)
        self.fc1 = nn.Linear(4800, 4800)
        self.fc2 = nn.Linear(4800, output_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x_size = 1
        for z in x.size():
            x_size *= z
        x_size /= x.size()[0]    
        x = x.view(x.size()[0], x_size) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
    
def predictAct(net, x, image_size):
    x = torch.from_numpy(x).cuda()
    x = x.view(1, image_size, image_size)
    x = Variable(x.float())
    return net(x)

def predict(net, x):
    x = torch.from_numpy(x).cuda()
    x = Variable(x.float())
    return net(x)
    
def experienceReplay(memory, replay_length, net, target_net,
                     discount, loss_func, step, image_size):
    if len(memory) < replay_length:
        batch_size = len(memory)
    else:
        batch_size = replay_length
    batch = sample(memory, batch_size)
    x_train = [mem[0] for mem in batch]
    q_future = [predictAct(target_net, mem[3], image_size) if mem[3] != 'T' else
                Variable(torch.from_numpy(np.zeros(output_size)).cuda()) for mem in batch]
    q = predict(net, np.asarray(x_train))
    q_target = q.data.cpu().numpy()
    for i in range(len(q_target)):
        q_target[i, batch[i][1]] = batch[i][2] + discount * torch.max(q_future[i].data) 
    q_target = Variable(torch.from_numpy(q_target).cuda(), requires_grad=False)    
    loss = loss_func(q, q_target)
    loss.backward(retain_variables=True)
    optimizer.step()
    optimizer.zero_grad()
    if step % 1000 == 0:
        target_net = copy.deepcopy(net)
    return loss.data.cpu().numpy()[0]

def processImage(image, size):
    image = np.sum(image, 2)
    image = imresize(image/3, (size,size))
    return image

def getObv(frames, skips, action):
    obv_stack = []
    sum_reward = 0
    for i in range(frames*skips):
        obs, reward, done, info = env.step(action)
        sum_reward += reward
        if done:
            obv_stack.append(obs)
            obv = np.sum(np.asarray(obv_stack), axis=0)
            break
        if i % skips == 0:
            obv_stack.append(obs)
            if len(obv_stack) == frames:
                obv = np.sum(np.asarray(obv_stack), axis=0) / len(obv_stack)
                break       
    return obv, sum_reward, done

def initNet():
    net = Net().cuda()
    net.zero_grad()
    return net

net = initNet()
target_net = copy.deepcopy(net)
loss_func = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

iterations = 10000 
memory = deque([])
memory_size = 1000000
image_size = 84
replay_length = 100
discount = 0.9
epsilon = 0.1
games_without_replay = 1000
frames = 4
skips = 2
decay = -.0005
step = 0
sumLoss = 0
x = []
y = []
totalReward = 0

for it in range(iterations):
    env.reset()
    obs, _, _ = getObv(frames, skips, env.action_space.sample())
    obs = processImage(obs, image_size)
    curr_step = step
    while True:
        step += 1
        obs_prev = copy.copy(obs)
        #env.render()
        if games_without_replay > it:
            action = env.action_space.sample()
        else:    
            action = np.argmax(predictAct(net, obs, image_size).data.cpu().numpy()[0])
            action = eGreedy(action, epsilon, step, decay, env)
        obs, reward, done = getObv(frames, skips, action)
        obs = processImage(obs, image_size)
        totalReward += reward
        if done:
            mem = (obs_prev, action, reward, 'T')
            addMem(memory, mem, memory_size)
            loss = experienceReplay(memory, replay_length, net, target_net,
                                    discount, loss_func, step, image_size)
            sumLoss += loss
            x.append(it)
            y.append(totalReward)
            print 'reward on iteration', it, ': ', totalReward
            totalReward = 0
            print 'loss on iteration', it, ': ', sumLoss/(step-curr_step)
            sumLoss = 0
            print '-------------------------------------------'
            break
        mem = (obs_prev, action, reward, obs)
        addMem(memory, mem, memory_size)
        loss = experienceReplay(memory, replay_length, net, target_net,
                                discount, loss_func, step, image_size)
        sumLoss += loss
plt.plot(x,y)            
plt.xlabel('iteration')
plt.ylabel('reward')
plt.grid(True)            
plt.show()
env.close()
