import gym
from gym import wrappers
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from random import sample

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
input_size = env.observation_space.sample().shape[0]
output_size = env.action_space.n

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
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
    
def predictAct(net, x):
    x = torch.from_numpy(x).cuda()
    x = x.view(1, input_size)
    x = Variable(x.float())
    return net(x)

def predict(net, x):
    x = torch.from_numpy(x).cuda()
    x = Variable(x.float())
    return net(x)
    
def experienceReplay(memory, replay_length, net, discount, loss_func):
    if len(memory) < replay_length:
        batch_size = len(memory)
    else:
        batch_size = replay_length
    batch = sample(memory, batch_size)
    x_train = [mem[0] for mem in batch]
    q_future = [predictAct(net, mem[3]) if mem[3] != 'T' else
                Variable(torch.from_numpy(np.zeros(output_size)).cuda()) for mem in batch]
    q = predict(net, np.asarray(x_train))
    q_target = q.data.cpu().numpy()
    for i in range(len(q_target)):
        q_target[i, batch[i][1]] = batch[i][2] + discount * torch.max(q_future[i].data) 
    q_target = Variable(torch.from_numpy(q_target).cuda(), requires_grad=False)    
    loss = loss_func(q, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.data.cpu().numpy()[0]

def initNet():
    net = Net().cuda()
    net.zero_grad()
    return net

net = initNet()
loss_func = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

iterations = 200 
memory = deque([])
memory_size = 10000
replay_length = 100
discount = 0.99
epsilon = 0.01
decay = -.001
step = 0

for it in range(iterations):
    obs = env.reset()
    while True:
        step += 1
        obs_prev = obs
        env.render()
        action = np.argmax(predictAct(net, obs).data.cpu().numpy()[0])
        action = eGreedy(action, epsilon, step, decay, env)
        obs, reward, done, info = env.step(action)
        if done:
            reward = -100
            mem = (obs_prev, action, reward, 'T')
            addMem(memory, mem, memory_size)
            loss = experienceReplay(memory, replay_length, net, discount, loss_func)
            break
        mem = (obs_prev, action, reward, obs)
        addMem(memory, mem, memory_size)
        loss = experienceReplay(memory, replay_length, net, discount, loss_func)
        if step % 100 == 0:
            print 'loss: ', loss
