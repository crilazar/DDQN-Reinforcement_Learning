# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:17:38 2019

@author: crila
"""
import os

import matplotlib.pyplot as plt 
import numpy as np

def plotLearning(x, scores, epsilons, filename):   
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-5):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)    
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1") 
    ax2.set_ylabel('Score', color="C1")       
    #ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)

from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
                Dense(fc1_dims, input_shape=(input_dims,)),
                Activation('relu'),
                Dense(fc2_dims),
                Activation('relu'),
                Dense(n_actions)])

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model

class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.99996,  epsilon_end=0.01,
                 mem_size=2000000, fname='ddqn_forex-b64-nn32_16-lr0_0005-env4.h5', replace_target=300):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 32, 16)
        self.q_target = build_dqn(alpha, n_actions, input_dims, 32, 16)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.q_target.model.set_weights(self.q_eval.model.get_weights())

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        #if self.epsilon == 0.0:
        self.update_network_parameters()

import gym
from gym import wrappers
import numpy as np
from env.forex_env_v4 import Forex1 as ForexTrading
from datetime import datetime
import shutil
import os
import os.path
from os import path

def write_to_log(message):
    with open("out-ddqn_forex-b64-nn32_16-lr0_0005-env4.log", "a") as file:
        time_to_print = datetime.now().strftime("%Y.%m %H:%M:%S")
        file.write(f"{time_to_print} : {message}\n")
        print(f"{time_to_print} : {message}")

if __name__ == '__main__':
    env = ForexTrading()
    write_to_log('---------------------------------------------')
    write_to_log('Environment loaded successfuly')
    ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=4, epsilon_dec=0.99996, epsilon=1.0, batch_size=64, input_dims=41)
    n_games = 1000    
    ddqn_scores = []
    eps_history = []
    
    #os.chdir('..')
    #source_load_file = 'gdrive/My Drive/RL_models/ddqn_model_forex1.h5'
    #dest_load_file = 'ddqn_model_forex1.h5'
    #shutil.copy2(source_load_file, dest_load_file)
    if path.exists("ddqn_forex-b64-nn32_16-lr0_0005-env4.h5"):
        ddqn_agent.load_model()
        write_to_log('---------------------------------------------')
        write_to_log('Previous learning model loaded')
    #env = wrappers.Monitor(env, "tmp/lunar-lander-ddqn-2",
    #                         video_callable=lambda episode_id: True, force=True)
    time_to_print = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    write_to_log(f'start training model: {time_to_print}')
    
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()
        eps_history.append(ddqn_agent.epsilon)

        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        time_to_print = datetime.now().strftime("%H:%M:%S")

        write_to_log(f'episode: {i} score: {int(score)} average score {int(avg_score)} time {time_to_print} balance {int(info[0])} buy {info[1]}/{info[2]} sell {info[3]}/{info[4]} pips_won {int(info[7])} pips_lost {int(info[8])} eps {eps_history[len(eps_history)-1]} avg_length {int(info[9])} min/max_length {int(info[10])}/{int(info[11])}')

        #if i % 2 == 0 and i > 0:
        ddqn_agent.save_model()
        source_save_file = 'ddqn_forex-b64-nn32_16-lr0_0005-env4.h5'
        #dest_save_file = 'ddqn_model_forex1_ep' + str(i) + '.h5'
        #shutil.copy2(source_save_file, dest_save_file)

    filename = 'forex1-ddqn.png'

    x = [i+1 for i in range(n_games)]
    plotLearning(x, ddqn_scores, eps_history, filename)

