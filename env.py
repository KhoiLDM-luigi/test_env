from gym import Env, spaces
import numpy as np
import pandas as pd
# from policy import QPolicy
from math import nan, log

from datetime import datetime 
import json

from enum import Enum

from agent import ActionMask

MAX_INT = 9223372036854775807

HOLD = 0
WAIT = 0
LONG = 1
SHORT = 2
SELL = 3

INVALID_PENALTY = 0
START_TRADE_REWARD = 5
WIN_REWARD = 10
LOSS_PENALTY = -10

class TradingStrategy(Enum):
    BUY_AND_HOLD=1
    BUY_SELL=2

class TradingEnv:
    def __init__(self):
        self.class_mapper = {
            TradingStrategy.BUY_AND_HOLD: BuyAndHold,
            TradingStrategy.BUY_SELL: BuySell
        }

        self.trading_strategy = TradingStrategy.BUY_AND_HOLD

    def set_strategy(self, trading_strategy: TradingStrategy):
        self.trading_strategy = trading_strategy

    def get_env(self, *args, **kwargs):
        return self.class_mapper[self.trading_strategy](*args, **kwargs)


class BuyAndHold(Env):
    def __init__(self, source: str, start_date: datetime, end_date: datetime, balance, trade_cost=0.01, render_mode=None, action_mask: ActionMask=None):
        super(BuyAndHold, self).__init__()
        # Environment information
        self.observation_space = spaces.Discrete(20, )
        self.action_space = spaces.Discrete(4,) # wait, long, short, sell
        assert render_mode == None or render_mode == "human" or render_mode == "plain_text_info", 'render mode must be \"human\" or None'
        self.render_mode = render_mode
        # Game data
        self.env = pd.read_csv(source)
    
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
        self.action_list = [WAIT, HOLD, LONG, SHORT, SELL] # WAIT = HOLD 
        self.invalid_action = []

        # Other
        self.action_mask = action_mask

    @property
    def max_step(self):
        return len(self.env.index)
    
    def step(self, action):
        current_price = self.env.loc[self.observation_point, 'close']
        terminated = False
        truncated = False
        reward = 0
        self.last_current_observation = self.observation_point
        try: # get next day use for step, if non available terminated the game
            next_observation_point = self.env.index[self.env.index.get_loc(self.observation_point) + 1] 
        except:
            next_observation_point = None
            terminated = True

        assert action in self.action_list, 'Action not in action space'

        if action in self.invalid_action:
            reward = INVALID_PENALTY
            self._info_update('invalid')
            
        elif action == HOLD:
            
            if self.last_action != WAIT: #Holding stock
                # changes, price = self._get_changes(current_price)
                # if self.last_action == LONG: reward = changes
                # else: reward = -changes
                
                if next_observation_point == None:
                    reward = 0
                else:
                    reward = self._buy_or_hold_reward(self.observation_point, next_observation_point)

                self.observation_point = next_observation_point # step
                self._info_update(action='Hold')
            else: #Not trading
                self.observation_point = next_observation_point # step
                reward = 0
                self._info_update(action='Wait')

        elif action == LONG: 
            self.observation_point = next_observation_point  # step
            self.play_balance -= current_price * (1 + self.trade_cost)
            self.entry_price = current_price
            self.last_action = LONG
            # changes, price = self._get_changes(current_price)

            if next_observation_point == None:
                reward = 0
            else:
                reward = START_TRADE_REWARD + self._buy_or_hold_reward(self.observation_point, next_observation_point)
            self.observation_point = next_observation_point  # step

            self._info_update(action='Start')

        elif action == SHORT:
            self.play_balance += current_price * (1 + self.trade_cost)
            self.entry_price = current_price
            self.last_action = SHORT
            # changes, price = self._get_changes(current_price)

            if next_observation_point == None:
                reward = 0
            else:
                reward = START_TRADE_REWARD + self._buy_or_hold_reward(self.observation_point, next_observation_point)
            self.observation_point = next_observation_point # step

            self._info_update(action='Start')
       
        else: # action == 4
            # Ending trade does not shift to next date invalid or not
            if self.last_action != WAIT: # Ending trade
                changes, price = self._get_changes(current_price)
               
                if self.last_action == LONG: 
                    self.play_balance += current_price * (1 + self.trade_cost)
                    is_win = changes >= 0   
                    
                else: # SHORT
                    self.play_balance -= current_price * (1 + self.trade_cost)
                    is_win = changes <= 0

                reward = self._sell_reward(is_win)
                self._info_update(sell=(self.last_action, is_win, price))

                self.last_action = WAIT
        
        # termination check
        if self.play_balance <= 0: 
            truncated = True

        # get observation
        if truncated or terminated:
            observation = self._get_observation(None)
            info = self._get_info()
            self.render(mode=self.render_mode)
        else:
            self._invalid_action_update()
            observation = self._get_observation(self.observation_point)
            info = {}

        return observation, reward, terminated or truncated, info
    
    def render(self, mode):
        if mode == "plain_text_info":
            print(json.dumps(self._get_info(), indent=4))
    
    def _invalid_action_update(self):
        if self.last_action != WAIT: # If transaction already started, additional transaction is allowed
            self.invalid_action = [LONG, SHORT]
        else:  # If there are no transaction, selling is not allowed
            self.invalid_action = [SELL]

        if self.action_mask != None:
            self.action_mask.update_invalid_action(self.invalid_action)
    
    def _get_changes(self, price): 
        return (price - self.entry_price) / self.entry_price, abs(price - self.entry_price)
    
    def _get_observation(self, observation_point):
        if observation_point == None:
            observation = np.zeros(20)
        else:
            try:
                start = self.env.index[self.env.index < observation_point][-19]
                observation = self.env.loc[start:self.observation_point, 'close'].to_numpy()
            except:
                observation = np.zeros(20)
                i = 1
                while True: 
                    try:
                        value = self.env.index[self.env.index < observation_point][-i]
                        observation[-i] =  self.env.loc[value, 'close']
                        i+=1
                    except: 
                        break
            
        return np.array([observation])

    def _portfolio_evaluation(self, observation_point):
        balance = self.balance 
        if self.last_action == LONG:
            assets_value = self.env.loc[observation_point, 'close'] * 1
        elif self.action_list == SHORT:
            assets_value = -self.env.loc[observation_point, 'close'] * 1
        else:
            assets_value = 0
        
        return balance + assets_value
    
    def _buy_or_hold_reward(self, observation_point, next_observation):
        return log(self._portfolio_evaluation(next_observation)) - log(self._portfolio_evaluation(observation_point))
    
    def _sell_reward(self, is_win):
        reward = (log(self.play_balance) - log(self._portfolio_evaluation(self.last_current_observation)))
        if is_win: 
            return 10 + reward
        return -10 + reward
    
    def _info_update(self, *args, **kwargs):
        if 'invalid' in args:
            self.total_invalid_action += 1
            return
        
        if 'action' in kwargs: # Starting transaction or Holding stock
            if kwargs['action'] == 'Start':
                # End of Wait period
                if self.wait_counter > self.longest_wait:
                    self.longest_wait = self.wait_counter
                if self.wait_counter < self.shortest_wait:
                    self.shortest_wait = self.wait_counter

                # New Holding period
                self.hold_counter = 1
                self.total_hold_count += 1
                self.total_hold_span += 1
                    
            elif kwargs['action'] == 'Hold': 
                # Holding period
                self.hold_counter += 1
                self.total_hold_span += 1

            elif kwargs['action'] == 'Wait':
                # Waiting period
                self.wait_counter += 1
                self.total_wait_span += 1

            self.survived_step += 1
            return
        
        if 'sell' in kwargs:
            action, is_win, price = kwargs['sell']
            if is_win: 
                if action == LONG: self.long_success += 1
                elif action == SHORT: self.short_success += 1
                self.total_gain += price
                if self.all_time_highest_gain < price:
                    self.all_time_highest_gain = price
            else:
                if action == LONG: self.long_failed += 1
                elif action == SHORT: self.short_failed += 1
                self.total_loss += price
                if self.all_time_highest_loss < price:
                    self.all_time_highest_loss = price

            if self.all_time_highest_balance < self.play_balance:
                self.all_time_highest_balance = self.play_balance
            # End of Holding period
            if self.longest_hold < self.hold_counter:
                self.longest_hold = self.hold_counter

            if self.shortest_hold > self.hold_counter:
                self.shortest_hold = self.hold_counter

            # New wait period
            self.wait_counter = 0
            self.total_wait_count += 1
            self.total_wait_span += 1

    def _get_info(self):
        try:
            average_gain = self.total_gain / (self.long_success + self.short_success)
        except:
            average_gain = nan

        try:
            average_loss = self.total_loss / (self.long_failed + self.short_failed)
        except:
            average_loss = nan

        try:
            average_wait = self.total_wait_span / self.total_wait_count
        except:
            average_wait = nan

        try:
            average_hold = self.total_hold_span / self.total_hold_count
        except:
            average_hold = nan

        return {
            # End of the game balance
            'final_balance': self.play_balance, 
            "highest_balance": self.all_time_highest_balance,
            # Action result
            "total_long_success": self.long_success,
            "total_long_failed": self.long_failed,
            "total_short_success": self.short_success,
            "total_short_failed": self.short_failed,   
            # Action behavior
            "total_invalid_action": self.total_invalid_action,
            "longest_hold": self.longest_hold,
            "shortest_hold": self.shortest_hold,
            "average_hold": average_hold,
            "longest_wait": self.longest_wait,
            "shortest_wait": self.shortest_hold,
            "average_wait": average_wait,
            # End of game profit/loss
            "highest_gain": self.all_time_highest_gain,
            "average_gain": average_gain,
            "highest_loss": self.all_time_highest_loss,
            "average_loss": average_loss,
            "survived_step": self.survived_step,            
        }
    
    def reset(self):
        # Reset game
        self.entry_price = 0
        self.last_action = 0 
        self.play_balance = self.balance
        self.observation_point = self.env.index[0]
        observation = self._get_observation(self.observation_point)
        self._invalid_action_update()

        # Play's evaluation
        self.total_invalid_action = 0
        self.long_success = 0
        self.short_success = 0
        self.long_failed = 0
        self.short_failed = 0
        self.survived_step = 0
        self.all_time_highest_balance = self.balance
        self.all_time_highest_gain = 0
        self.all_time_highest_loss = 0

        self.longest_hold = 0
        self.shortest_hold = MAX_INT
        self.longest_wait = 0
        self.shortest_wait = MAX_INT
        # average profit/loss
        self.total_loss = 0
        self.total_gain = 0
        # average hold
        self.total_hold_count = 0
        self.total_hold_span = 0
        # average wait
        self.total_wait_count = 1
        self.total_wait_span = 0
        # other 
        self.wait_counter = 0
        self.hold_counter = 0
       
        return observation
    

class BuySell(Env):
    def __init__(self, source: str, start_date: datetime, end_date: datetime, balance, trade_cost=0.01, render_mode=None):
        super(BuySell, self).__init__()
        # Environment information
        self.observation_space = spaces.Discrete(20, )
        self.action_space = spaces.Discrete(3,) # hold/wait transaction, long, short, sell
        assert render_mode == None or render_mode == "human" or render_mode == "plain_text_info", 'render mode must be \"human\" or None'
        self.render_mode = render_mode
        # Game data
        self.env = pd.read_csv(source)
    
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
        self.action_list = [HOLD, LONG, SHORT] # WAIT = HOLD 

    @property
    def max_step(self):
        return len(self.env.index)
    
    def step(self, action):
        terminated = False
        truncated = False
        reward = 0
        try: # get next day use for step, if non available terminated the game
            next_observation_point = self.env.index[self.env.index.get_loc(self.observation_point) + 1] 
        except:
            next_observation_point = None
            terminated = True

        assert action in self.action_list, 'Action not in action space'

        if not terminated:
            if action == WAIT:
                reward = 0
                self._info_update(state=(WAIT, None, None))

            elif action == LONG: 
                changes, price = self._get_changes()
                reward = changes
                self.play_balance = self.play_balance + price  - abs(price) * self.trade_cost
                is_win = changes >= 0
                self._info_update(state=(LONG, is_win, price))

            elif action == SHORT:
                changes, price = self._get_changes()
                reward = -changes
                self.play_balance = self.play_balance - price - abs(price) * self.trade_cost
                is_win = changes <= 0
                self._info_update(state=(SHORT, is_win, price))
        
        # termination check
        if self.play_balance <= 0: 
            truncated = True

        self.observation_point = next_observation_point  # step

        # get observation
        if truncated or terminated:
            observation = np.array([np.zeros(21)])
            info = self._get_info()
            print("Play end:")
            self.render(mode=self.render_mode)
        else:
            observation = self._get_observation()
            info = {}

        return observation, reward, terminated or truncated, info

    
    def _get_changes(self): 
        data = self.env.loc[self.observation_point]
        return (data['close'] - data['open']) / data['open'], (data['close'] - data['open'])

    def _get_observation(self):
        if self.observation_point == None:
            observation = self.env.loc[-20:-1, 'close'].to_numpy()
        else:
            try:
                start = self.env.index[self.env.index < self.observation_point][-19]
                open = self.env.loc[start:self.observation_point, 'open']
                close = self.env.loc[start:self.observation_point, 'close']
                observation = ((close - open) * open).to_numpy()
            except:
                observation = np.zeros(20)
                i = 1
                while True: 
                    try:
                        value = self.env.index[self.env.index < self.observation_point][-i]
                        observation[-i] =  (self.env.loc[value, 'close'] - self.env.loc[value, 'open']) / self.env.loc[value, 'open']
                        i+=1
                    except: 
                        break
        observation = np.append(observation, self.env.loc[self.observation_point, 'open'])
        return np.array([observation])
    
    def render(self, mode):
        if mode == "plain_text_info":
            print(json.dumps(self._get_info(), indent=4))
    
    def _info_update(self, state):
        action, is_win, price = state
      
        if action == WAIT:
            # Waiting period
            self.wait_counter += 1
            self.total_wait_span += 1
            self.survived_step += 1
        elif action == LONG or action == SHORT:
            price = abs(price)

            # End of Wait period
            if self.wait_counter > self.longest_wait:
                self.longest_wait = self.wait_counter
            if self.wait_counter < self.shortest_wait:
                self.shortest_wait = self.wait_counter

            if is_win:
                if action == LONG: self.long_success += 1
                elif action == SHORT: self.short_success += 1
                self.total_gain += price
                if self.all_time_highest_gain < price:
                    self.all_time_highest_gain = price
            else:
                if action == LONG: self.long_failed += 1
                elif action == SHORT: self.short_failed += 1
                self.total_loss += price
                if self.all_time_highest_loss < price:
                    self.all_time_highest_loss = price

            if self.all_time_highest_balance < self.play_balance:
                self.all_time_highest_balance = self.play_balance
      
            # New wait period
            self.wait_counter = 0
            self.total_wait_count += 1
            self.total_wait_span += 1

            self.survived_step+=1


    def _get_info(self):
        try:
            average_gain = self.total_gain / (self.long_success + self.short_success)
        except:
            average_gain = nan

        try:
            average_loss = self.total_loss / (self.long_failed + self.short_failed)
        except:
            average_loss = nan

        try:
            average_wait = self.total_wait_span / self.total_wait_count
        except:
            average_wait = nan

        return {
            # End of the game balance
            'final_balance': self.play_balance, 
            "highest_balance": self.all_time_highest_balance,
            # Action result
            "total_long_success": self.long_success,
            "total_long_failed": self.long_failed,
            "total_short_success": self.short_success,
            "total_short_failed": self.short_failed,   
            # Action behavior
            "longest_wait": self.longest_wait,
            "shortest_wait": self.shortest_wait,
            "average_wait": average_wait,
            # End of game profit/loss
            "highest_gain": self.all_time_highest_gain,
            "average_gain": average_gain,
            "highest_loss": self.all_time_highest_loss,
            "average_loss": average_loss,
            "survived_step": self.survived_step,            
        }
    
    def reset(self):
        # Reset game
        self.entry_price = 0
        self.last_action = 0 
        self.play_balance = self.balance
        self.observation_point = self.env.index[0]
        observation = self._get_observation()

        # Play's evaluation
        self.long_success = 0
        self.short_success = 0
        self.long_failed = 0
        self.short_failed = 0
        self.survived_step = 0
        self.all_time_highest_balance = self.balance
        self.all_time_highest_gain = 0
        self.all_time_highest_loss = 0

        self.longest_wait = 0
        self.shortest_wait = MAX_INT
        # average profit/loss
        self.total_loss = 0
        self.total_gain = 0
        # average hold
        self.total_hold_count = 0
        self.total_hold_span = 0
        # average wait
        self.total_wait_count = 1
        self.total_wait_span = 0
        # other 
        self.wait_counter = 0
        self.hold_counter = 0
       
        return observation