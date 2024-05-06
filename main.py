#Keras library to define the NN to be used
from keras.models import Sequential
#Layers used in the NN considered
from keras.layers import Dense, Activation, Flatten
#Activation Layers used in the source code
from keras.layers import LeakyReLU

from datetime import datetime

from player import Trader, TraderWithActionMasking
from env import TradingEnv, TradingStrategy
from agent import ActionMask
from stockenv import BuyAndHold

model = Sequential()
model.add(Flatten(input_shape=(1,1,20)))
model.add(Dense(32,activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(64, activation='linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(3))
model.add(Activation('linear'))


trading_env = TradingEnv()
trading_env.set_strategy(TradingStrategy.BUY_AND_HOLD)

action_mask = ActionMask(4)

trader = Trader(model, 3)

train_start = datetime(2000, 1, 3, 9, 0, 0)
train_end = datetime(2008, 12, 30, 14, 0 , 0)
train_env = trading_env.get_env('data/dax.csv', train_start, train_end, 10000, 0.01, render_mode="plain_text_info", action_mask=action_mask)
# train_env = BuyAndHold(data_source='data/dax.csv', start_date=train_start, end_date=train_end, balance=10000, render_mode="plain_text_info")

test_start = datetime(2009, 1, 2, 8, 0, 0)
test_end = datetime(2010, 12, 30, 14, 0 , 0)
test_env = trading_env.get_env('data/dax.csv', train_start, test_end, 10000, 0.01, render_mode="plain_text_info", action_mask=action_mask)
# test_env = BuyAndHold(data_source='data/dax.csv', start_date=train_start, end_date=test_end, balance=10000, render_mode="plain_text_info")

trader.train(train_env, 50)
trader.save_checkpoint('weight.pt')

trader.load_checkpoint('weight.pt')
trader.play(test_env, 1)