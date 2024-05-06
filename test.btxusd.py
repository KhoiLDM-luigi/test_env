#Keras library to define the NN to be used
from keras.models import Sequential
#Layers used in the NN considered
from keras.layers import Dense, Activation, Flatten
#Activation Layers used in the source code
from keras.layers import LeakyReLU

from datetime import datetime

from player import Trader
from env import TradingEnv, TradingStrategy

model = Sequential()
model.add(Flatten(input_shape=(1,1,21)))
model.add(Dense(32,activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(64, activation='linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(3))
model.add(Activation('linear'))

trader = Trader(model, 3)

trading_env = TradingEnv()
trading_env.set_strategy(TradingStrategy.BUY_SELL)

test_start = datetime(2022, 2, 5, 0, 0, 0)
test_end = datetime(2024, 1, 25, 23, 0 , 0)
test_env = trading_env.get_env('data/BTX-USD.csv', test_start, test_end, 10000, 0.01, render_mode="plain_text_info")

trader.load_checkpoint('weight.pt')
trader.train(test_env, 10) # warm up
trader.play(test_env, 1)