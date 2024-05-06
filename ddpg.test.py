#Keras library to define the NN to be used
from keras.models import Sequential, Model
#Layers used in the NN considered
from keras.layers import Dense, Activation, Flatten, Concatenate, Input
#Activation Layers used in the source code
from keras.layers import LeakyReLU
from keras.optimizers import Adam

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


from datetime import datetime

import numpy as np

from stockenv import BuyAndHold


class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)



train_start = datetime(2000, 1, 3, 9, 0, 0)
train_end = datetime(2008, 12, 30, 14, 0 , 0)
env = BuyAndHold(data_source='data/dax.csv', start_date=train_start, end_date=train_end, balance=10000, render_mode="plain_text_info")

nb_actions = 3

actor = Sequential()
actor.add(Flatten(input_shape=(1,1,20)))
actor.add(Dense(32,activation='linear'))
actor.add(Dense(128, activation='linear'))
actor.add(Dense(64, activation='linear'))
actor.add(LeakyReLU(alpha=.001))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1, 1, 20), name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Dense(400)(flattened_observation)
x = Activation('relu')(x)
x = Concatenate()([x, action_input])
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  processor=MujocoProcessor())
agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

agent.fit(env, nb_steps=1000000, visualize=False, verbose=1)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)