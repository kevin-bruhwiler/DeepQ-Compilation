import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from random import sample
import sys

env = gym.make('Pendulum-v0')
env = wrappers.Monitor(env, '/tmp/pendulum-experiment-1', force=True)
input_size = env.observation_space.sample().shape[0]
output_size = env.action_space.sample().shape[0]

class Actor():
    def __init__(self, input_size, output_size):
        self.y_ = tf.placeholder(dtype=tf.float32)
        self.x = tf.placeholder(dtype=tf.float32)
        self.w1, y = self.linear(self.x, input_size, 50)
        self.h1 = tf.nn.relu(y)
        self.w2, y = self.linear(self.h1, 50, 50)
        self.h2 = tf.nn.relu(y)
        self.w3, y = self.linear(self.h2, 50, output_size)
        self.y = tf.nn.tanh(y)
        self.loss = tf.losses.mean_squared_error(self.y_, self.y)
        self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        return

    def linear(self, x, dim1, dim2):
        W = tf.Variable(tf.truncated_normal([dim1, dim2], stddev=0.1))
        b = tf.Variable(tf.constant(1.0), [1,dim2]) 
        return W, tf.matmul(x+b,W)

    def get_sess(self):
        return self.sess

class Critic():
    def __init__(self, input_size, output_size):
        self.a = tf.placeholder(dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32)
        self.x = tf.placeholder(dtype=tf.float32)
        self.w1, y = self.linear(self.x, input_size, 50)
        self.h1 = tf.nn.relu(y)
        self.w2, y = self.linear(self.h1, 50, 50)
        self.h2 = tf.nn.relu(y)
        #add a to final layer
        self.w3, self.y = self.linear(tf.concat([self.a, self.h2],1), 50+output_size, 1)
        self.loss = tf.losses.mean_squared_error(self.y_, self.y)
        self.a_grad = []
        self.a_grad.append(tf.gradients(self.loss, [self.a])[0])
        self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        return

    def linear(self, x, dim1, dim2):
        W = tf.Variable(tf.truncated_normal([dim1, dim2], stddev=0.1))
        b = tf.Variable(tf.constant(1.0), [1,dim2]) 
        return W, tf.matmul(x+b,W)

    def get_sess(self):
        return self.sess

    def get_a_grad(self):
        return self.a_grad

def copyNet(curr, other):
    curr.w1 = tf.Variable(other.w1.initialized_value())
    curr.w2 = tf.Variable(other.w2.initialized_value())
    curr.w3 = tf.Variable(other.w3.initialized_value())
    return

def updateNet(curr, other):
    tau = 0.001
    curr.w1 = tf.Variable(tau*curr.w1.initialized_value() + (1-tau)*curr.w1.initialized_value())
    curr.w2 = tf.Variable(tau*curr.w1.initialized_value() + (1-tau)*curr.w1.initialized_value())
    curr.w3 = tf.Variable(tau*curr.w1.initialized_value() + (1-tau)*curr.w1.initialized_value())
    return

def addMem(memory, mem, memory_size):
    if len(memory) >= memory_size:
        memory.popleft()
    memory.append(mem)
    return

def noise(action, it):
    return np.clip(action + np.random.uniform(-1,1)/(1.+it), -1, 1)
    
def predictAct(actor, sess, x):
    return sess.run(actor.y, {actor.x:x.reshape((1,x.shape[0]))})[0]

def predictActs(actor, sess, x):
    return sess.run(actor.y, {actor.x:x})

def critiqueOne(critic, sess, x, a):
    return sess.run(critic.y, {critic.x:x.reshape((1,x.shape[0])), critic.a:a.reshape((1,a.shape[0]))})[0]

def critique(critic, sess, x, a):
    return sess.run(critic.y, {critic.x:x, critic.a:a})
    
def experienceReplay(memory, replay_length, actor, actor_sess,
                     target_actor, target_actor_sess,critic, critic_sess,
                     target_critic, target_critic_sess, discount, step):
    if len(memory) < replay_length:
        batch_size = len(memory)
    else:
        batch_size = replay_length
    batch = sample(memory, batch_size)
    x_train = [mem[0] for mem in batch]
    acts = [mem[1] for mem in batch]
    zero = np.asarray([0] * output_size)
    q_future = [critiqueOne(target_critic, target_critic_sess, mem[3],
                            predictAct(target_actor, target_actor_sess, mem[3]))
                if mem[3] != 'T' else zero for mem in batch]
    q_target = []
    for i in range(len(q_future)):
        q_target.append(batch[i][2] + discount * q_future[i])
    _, l = critic_sess.run([critic.train,critic.loss], {critic.a:acts, critic.x:x_train, critic.y_:q_target})
    a_out = predictActs(actor, actor_sess, x_train)
    a_grads = target_critic_sess.run(target_critic.a_grad, {target_critic.x:x_train,
                                                            target_critic.y_:q_target,
                                                            target_critic.a:a_out})[0]
    _, pl = actor_sess.run([actor.train,actor.loss], {actor.x:x_train, actor.y_:a_grads})
    updateNet(target_actor, actor)
    updateNet(target_critic, critic)
    return l, pl


actor = Actor(input_size, output_size)
target_actor = Actor(input_size, output_size)
copyNet(target_actor, actor)
critic = Critic(input_size, output_size)
target_critic = Critic(input_size, output_size)
copyNet(target_critic, critic)
actor_sess = actor.get_sess()
critic_sess = critic.get_sess()
target_actor_sess = target_actor.get_sess()
target_critic_sess = target_critic.get_sess()

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
        action = predictAct(actor, actor_sess, obs)
        action = noise(action, it)
        obs, reward, done, info = env.step(action)
        obs = obs.reshape(3)
        totalReward += reward
        if done:
            mem = (obs_prev, action, reward, 'T')
            addMem(memory, mem, memory_size)
            loss, p_loss = experienceReplay(memory, replay_length, actor, actor_sess,
                                            target_actor, target_actor_sess,critic, critic_sess,
                                            target_critic, target_critic_sess, discount, step)
            sumLoss += loss
            p_sumLoss += p_loss
            x.append(it)
            y.append(totalReward)
            print ('reward on iteration', it, ': ', totalReward)
            totalReward = 0
            print ('average loss on iteration', it, ': ', sumLoss/(step-curr_step))
            sumLoss = 0
            print ('average p_loss on iteration', it, ': ', p_sumLoss/(step-curr_step))
            p_sumLoss = 0
            print ('--------------------------------------------------------')
            break
        mem = (obs_prev, action, reward, obs)
        addMem(memory, mem, memory_size)
        loss, p_loss = experienceReplay(memory, replay_length, actor, actor_sess,
                                        target_actor, target_actor_sess,critic, critic_sess,
                                        target_critic, target_critic_sess, discount, step)
        sumLoss += loss
        p_sumLoss += p_loss
plt.plot(x,y)            
plt.xlabel('iteration')
plt.ylabel('reward')
plt.grid(True)            
plt.show()
