from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

from keras.optimizers import Adam
from env import TradingEnv

import pandas as pd

from agent import CustomDQNAgent

class DDPGTrader:
    def __init__(self, model, nb_actions):
        self.nb_actions = nb_actions
        self.model = model

        self.policy = EpsGreedyQPolicy()
        self.memory = SequentialMemory(limit=100000, window_length=1)

        self.agent = DQNAgent(model=self.model, policy=self.policy, nb_actions=self.nb_actions, memory=self.memory, 
                              nb_steps_warmup=200, target_model_update=1e-1, 
                              enable_double_dqn=True, enable_dueling_network=True)
        
        self.agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    def train(self, env: TradingEnv, epoch):
        # expected train step: highest possible step per play * train epoch
        # actual epoch during training might be larger than the train epoch value
        max_step = epoch * env.max_step 
        self.agent.fit(env, nb_steps=max_step)

    def save_checkpoint(self, destination):
        self.agent.save_weights(destination, overwrite=True)
    
    def load_checkpoint(self, source):
        self.agent.load_weights(source)
    
    def play(self, env: TradingEnv, epoch=1):
        self.agent.test(env, nb_episodes=epoch)

class TraderWithActionMasking:
    def __init__(self, model, nb_actions, action_mask):
        self.nb_actions = nb_actions
        self.model = model

        self.policy = EpsGreedyQPolicy()
        self.memory = SequentialMemory(limit=100000, window_length=1)

        self.agent = CustomDQNAgent(action_mask=action_mask, model=self.model, policy=self.policy, nb_actions=self.nb_actions, memory=self.memory, 
                              nb_steps_warmup=200, target_model_update=1e-1, 
                              enable_double_dqn=True, enable_dueling_network=True)
        
        self.agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    def train(self, env: TradingEnv, epoch):
        # expected train step: highest possible step per play * train epoch
        # actual epoch during training might be larger than the train epoch value
        max_step = epoch * env.max_step 
        self.agent.fit(env, nb_steps=max_step)

    def save_checkpoint(self, destination):
        self.agent.save_weights(destination, overwrite=True)
    
    def load_checkpoint(self, source):
        self.agent.load_weights(source)
    
    def play(self, env: TradingEnv, epoch=1):
        self.agent.test(env, nb_episodes=epoch)

class Trader:
    def __init__(self, model, nb_actions):
        self.nb_actions = nb_actions
        self.model = model

        self.policy = EpsGreedyQPolicy()
        self.memory = SequentialMemory(limit=100000, window_length=1)

        self.agent = DQNAgent(model=self.model, policy=self.policy, nb_actions=self.nb_actions, memory=self.memory, 
                              nb_steps_warmup=200, target_model_update=1e-1, 
                              enable_double_dqn=True, enable_dueling_network=True)
        
        self.agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    def train(self, env: TradingEnv, epoch):
        # expected train step: highest possible step per play * train epoch
        # actual epoch during training might be larger than the train epoch value
        max_step = epoch * env.max_step 
        self.agent.fit(env, nb_steps=max_step)

    def save_checkpoint(self, destination):
        self.agent.save_weights(destination, overwrite=True)
    
    def load_checkpoint(self, source):
        self.agent.load_weights(source)
    
    def play(self, env: TradingEnv, epoch=1):
        self.agent.test(env, nb_episodes=epoch)

class TraderWithActionMasking:
    def __init__(self, model, nb_actions, action_mask):
        self.nb_actions = nb_actions
        self.model = model

        self.policy = EpsGreedyQPolicy()
        self.memory = SequentialMemory(limit=100000, window_length=1)

        self.agent = CustomDQNAgent(action_mask=action_mask, model=self.model, policy=self.policy, nb_actions=self.nb_actions, memory=self.memory, 
                              nb_steps_warmup=200, target_model_update=1e-1, 
                              enable_double_dqn=True, enable_dueling_network=True)
        
        self.agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    def train(self, env: TradingEnv, epoch):
        # expected train step: highest possible step per play * train epoch
        # actual epoch during training might be larger than the train epoch value
        max_step = epoch * env.max_step 
        self.agent.fit(env, nb_steps=max_step)

    def save_checkpoint(self, destination):
        self.agent.save_weights(destination, overwrite=True)
    
    def load_checkpoint(self, source):
        self.agent.load_weights(source)
    
    def play(self, env: TradingEnv, epoch=1):
        self.agent.test(env, nb_episodes=epoch)