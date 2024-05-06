from gym import Env, spaces
import numpy as np
import pandas as pd
# from policy import QPolicy
from math import nan, log

from datetime import datetime 
import json

from enum import Enum

from agent import ActionMask

# Constant
HOLD = 0
WAIT = 0 
BUY = 1
SELL = 2

PENALTY = -10

class BuyAndHold(Env):
    def __init__(self, data_source: str, start_date: datetime, end_date: datetime, 
                 go_short=False, balance=10000, trade_cost=0.01, 
                 render_mode=None) -> None:
        
        open('log.txt', 'w').close()
        
        super(BuyAndHold, self).__init__()
        # Environment information
        self.observation_space = spaces.Discrete(20, )
        self.action_space = spaces.Discrete(3,) # wait, long, short, sell
        assert render_mode == None or render_mode == "human" or render_mode == "plain_text_info", 'render mode must be \"human\" or None'
        self.render_mode = render_mode
        # Game data
        self.env = pd.read_csv(data_source)
    
        self.env.columns = self.env.columns.str.lower()
        if 'datetime' in self.env.columns:
            self.env['datetime'] = pd.to_datetime(self.env['datetime'], format='%m/%d/%Y %H:%M')
        else:
            assert 'date' in self.env.columns and 'time' in self.env.columns, 'Dataset required both Date and Time column'
            self.env['datetime'] = pd.to_datetime(self.env['date'] + ' ' + self.env['time'], format='%m/%d/%Y %H:%M')
            self.env.drop(columns=['date', 'time'], inplace=True, axis=1)

        self.env.set_index('datetime', inplace=True)
        self.env = self.env.loc[start_date : end_date]

        # Game criterial and effect
        self.balance = balance
        self.trade_cost = trade_cost
        self.go_short = go_short
        self.action_list = [HOLD, WAIT, BUY, SELL]

        self.short_hold = 0
        self.long_hold = 0

    def _portfolio_value(self, observation_point): 
        if observation_point == None: 
            return self.play_balance
        
        observation_price = self.long_hold * self.env.loc[observation_point, 'close']
        
        long_value = self.long_hold * observation_price
        value = self.play_balance + long_value
        if self.go_short:
            short_value = self.short_hold * observation_price
            value -= short_value

        return value

    def _get_observation(self, observation_point):
        if observation_point == None:
            observation = self.env['close'][-20:-1].to_numpy()
        else:
            try:
                start = self.env.index[self.env.index < observation_point][-19]
                open = self.env.loc[start:observation_point, 'open']
                close = self.env.loc[start:observation_point, 'close']
                observation = ((close - open) * open).to_numpy()
            except:
                observation = np.zeros(20)
                i = 1
                while True: 
                    try:
                        value = self.env.index[self.env.index < observation_point][-i]
                        observation[-i] =  (self.env.loc[value, 'close'] - self.env.loc[value, 'open']) / self.env.loc[value, 'open']
                        i+=1
                    except: 
                        break
        # observation = np.append(observation, self.env.loc[observation_point, 'open'])
        return np.array([observation])
    
    @property
    def max_step(self):
        return len(self.env.index)
    
    def action_encoder(self, action):
        return np.argmax(action)
    
    def step(self, action): 
        action = self.action_encoder(action)
        assert action in self.action_list, 'Incorrect action'

        with open('log.txt', 'a') as f: f.write(f"{action}\n")

        current_price = self.env.loc[self.observation_point, 'close']
        terminated = False
        truncated = False
        reward = 0


        last_portfolio_value = self._portfolio_value(self.last_observation_point)
        curr_portfolio_value = self._portfolio_value(self.observation_point)

        GENERIC_REWARD = (curr_portfolio_value - last_portfolio_value) / last_portfolio_value
        self.last_observation_point = self.observation_point

        try: # get next day use for step, if non available terminated the game
            next_observation_point = self.env.index[self.env.index.get_loc(self.observation_point) + 1] 
        except:
            next_observation_point = None
            terminated = True

        if action == HOLD:
            if self.last_action != WAIT: # BUY or SELL
                reward = GENERIC_REWARD

                 # step 
                self.survived_step += 1
                self.observation_point = next_observation_point

            else:
                reward = 0

                 # step 
                self.survived_step += 1
                self.observation_point = next_observation_point

        self.last_action = action
        
        if action == BUY:
            if self.go_short and self.short_hold > 0: # return "borrowed" hold
                returned_hold_price = self.short_hold * current_price
                self.play_balance = self.play_balance - returned_hold_price - returned_hold_price * self.trade_cost
                self.short_hold = 0
                reward = GENERIC_REWARD
                self.last_action = WAIT

                if reward >= 0:
                        self.wins += 1
                else:
                    self.loss += 1 

                # step 
                self.survived_step += 1
                self.observation_point = next_observation_point
 
            else: # buy new hold 
                self.long_hold += 1
                self.play_balance = self.play_balance - current_price - current_price * self.trade_cost
                reward = GENERIC_REWARD

                # step 
                self.survived_step += 1
                self.observation_point = next_observation_point

        if action == SELL:
            if self.go_short: # "borrowed" hold to sell
                self.short_hold += 1
                self.play_balance = self.play_balance + current_price - current_price * self.trade_cost
                reward = GENERIC_REWARD

                # step 
                self.survived_step += 1
                self.observation_point = next_observation_point

            else: # sell long hold
                if self.long_hold <= 0: # didn't hold any during sell
                    reward = PENALTY
                else:
                    sell_hold_price = self.long_hold * current_price
                    self.play_balance = self.play_balance + sell_hold_price - sell_hold_price * self.trade_cost
                    self.long_hold = 0
                    reward = GENERIC_REWARD
                    self.last_action = WAIT

                    if reward >= 0:
                        self.wins += 1
                    else:
                        self.loss += 1 

                    # step 
                    self.survived_step += 1
                    self.observation_point = next_observation_point

        if self._portfolio_value(self.last_observation_point) < 0: # portfolio value is in negative
            truncated = True
        
        if truncated or terminated:
            self.info = {
                "final_balance": self.play_balance,
                "portfolio_value": self._portfolio_value(self.last_observation_point),
                "wins": self.wins,
                "loss": self.loss ,
                "survived_step": self.survived_step,
            }

            self.render(self.render_mode)
        else:
            self.info = {}

        observation = self._get_observation(self.observation_point)

        return observation, reward, terminated or truncated, self.info

    def render(self, mode):
        if mode == "plain_text_info":
            print(json.dumps(self.info, indent=4))
    
    def reset(self):
        # Reset game
        self.entry_price = 0
        self.last_action = WAIT
        self.play_balance = self.balance
        self.observation_point = self.env.index[0]
        self.last_observation_point = None
        observation = self._get_observation(self.observation_point)

        # Play's evaluation
        self.survived_step = 0
        self.wins = 0
        self.loss = 0
        self.invalid = 0

        return observation
