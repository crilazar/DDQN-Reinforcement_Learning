import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

INITIAL_ACCOUNT_BALANCE = 5000


class Forex1(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Forex1, self).__init__()

        df = pd.read_csv('./data/GBPUSD_2019_full_year_M5.csv')
        df = df.sort_values('Date_time')
        self.df = df.iloc[:, 1:].values
        self.max_step = len(self.df) - 2

        self.CurrentMarketLevel = 0
        self.active_trade = 0
        self.profit = 0
        self.profitable_buy = 0
        self.profitable_sell = 0
        self.notprofitable_buy = 0
        self.notprofitable_sell = 0
        self.trade_length = 0
        self.last_trade_length = 0
        self.pips_won = 0
        self.pips_lost = 0
        self.avg_length = [0]

        self.account_balance = INITIAL_ACCOUNT_BALANCE
        self.before_trade_acount_balance = self.account_balance

        # Actions of the format hold, Buy, Sell, close
        self.action_space = spaces.Discrete(4)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(22,), dtype=np.float16)

    def _get_current_step_data(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        data_current_step = self.df[self.current_step]
        self.CurrentMarketLevel = data_current_step[19]
        data_current_step = data_current_step[:-1]      # remove current market data = last element
        self._calculate_profit()
        output_data = np.append(data_current_step, [[
                    self.active_trade,
                    self.trade_length,
                    float(self.profit)
                ]])
        obs = self._normalize_data(output_data)
        obs = obs.reshape(22,)
        return obs

    def _normalize_data(self, input_data):
        norm_data = input_data
        norm_data[0] = input_data[0] / 100                    # Cycle_12_14_H4_0
        norm_data[1] = input_data[1] / 100                    # Cycle_12_14_H4_2
        norm_data[2] = input_data[2] / 100                    # Cycle_12_14_H1_0
        norm_data[3] = input_data[3] / 100                    # Cycle_12_14_H1_2
        norm_data[4] = input_data[4] / 100                    # Cycle_12_14_M15_0
        norm_data[5] = input_data[5] / 100                    # Cycle_12_14_M15_2
        norm_data[6] = input_data[6] / 100                    # Cycle_12_14_M5_0
        norm_data[7] = input_data[7] / 100                    # Cycle_12_14_M5_2       
        norm_data[8] = input_data[8] / 100                    # ColorStochastic_M5
        norm_data[9] = input_data[9] / 100                    # ColorStochastic_M15
        norm_data[10] = input_data[10]                        # LagN_blue
        norm_data[11] = input_data[11]                        # LagN_purple
        norm_data[12] = (input_data[12] + 1) / 2              # LagN_yellow_extreme
        norm_data[13] = (input_data[13] + 1) / 2              # EMA_trend_M15
        norm_data[14] = input_data[14] / 500                  # Market_to_EMA_blue
        norm_data[15] = (input_data[15] + 1) / 2              # EMA_trend_H1
        norm_data[16] = input_data[16] / 6000                 # EMAage_H1
        norm_data[17] = input_data[17] / 300                  # EMAdelta_H1
        norm_data[18] = input_data[18] / 500                  # Market_to_EMA_blue_H1
        norm_data[19] = input_data[19] / 2                    # active_trade
        norm_data[20] = input_data[20] / 1000                 # trade length        
        norm_data[21] = (input_data[21] + 1000) / 2000        # profit

        return norm_data

    def _close_trade(self):
        self.close_profit = self.profit
        self.account_balance = self.before_trade_acount_balance + self.close_profit

        if self.active_trade == 1:
            if self.close_profit > 4:
                self.profitable_buy += 1
                self.pips_won += self.close_profit
                self.avg_length.append(self.trade_length)
            else:
                self.notprofitable_buy += 1
                self.pips_lost -= self.close_profit
        if self.active_trade == 2:
            if self.close_profit > 4:
                self.profitable_sell += 1
                self.pips_won += self.close_profit
                self.avg_length.append(self.trade_length)
            else:
                self.notprofitable_sell += 1
                self.pips_lost -= self.close_profit

        self.profit = 0
        self.active_trade = 0
        self.trade_open_price = 0
        self.before_trade_acount_balance = 0
        self.last_trade_length = self.trade_length
        self.trade_length = 0

    def _take_action(self, action):
        action_type = action
        
        if self.active_trade != 0:
            self.trade_length += 1

        if action_type == 1 and self.active_trade != 1:       # Buy trade action when active_trade = 1
            if self.active_trade == 2:
                self._close_trade()
            self.active_trade = 1
            self.trade_length = 0
            self.trade_open_price = self.CurrentMarketLevel
            self.before_trade_acount_balance = self.account_balance
            self.profit = 0

        if action_type == 2 and self.active_trade != 2:     # Sell trade action when active_trade = 2
            if self.active_trade == 1:
                self._close_trade()
            self.active_trade = 2
            self.trade_length = 0
            self.trade_open_price = self.CurrentMarketLevel
            self.before_trade_acount_balance = self.account_balance
            self.profit = 0

        if action_type == 3 and self.active_trade != 0:      # Close trade action
            self._close_trade()
            
        #print(f'action_type = {action} and active_trade = {self.active_trade}')
        
    def _calculate_profit(self):
        
        if self.active_trade != 0:               # calculate profit only if any active trade
            if self.active_trade == 2:
                self.profit = (self.trade_open_price - self.CurrentMarketLevel) * 10000
            if self.active_trade == 1:
                self.profit = (self.CurrentMarketLevel - self.trade_open_price) * 10000
            #self.account_balance = self.before_trade_acount_balance  + self.profit
        
        if float(self.profit) < -100:              # close active trade if profit less than -100
            self._close_trade()
            
    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        reward = 0
        done = float(self.account_balance) <= 0
        done = self.current_step > self.max_step -1
        
        obs = self._get_current_step_data()

        info = [float(self.account_balance), self.profitable_buy, self.notprofitable_buy, self.profitable_sell,\
            self.notprofitable_sell, self.trade_length, self.last_trade_length, self.pips_won, self.pips_lost, int(np.mean(self.avg_length)), np.min(self.avg_length), np.max(self.avg_length)]

        if self.trade_length > 0:
            if self.profit > 0:
                reward = self.trade_length / 100 + self.profit / 100
            elif self.profit > -30:
                reward = self.trade_length / 500
            else:
                reward = 0

        # bonus for closing a positive trade
        if self.active_trade == 0 and self.close_profit != 0:
            if self.close_profit > 100:
                reward = self.close_profit + 10 + self.last_trade_length / 2
            elif self.close_profit > 50:
                reward = self.close_profit + self.last_trade_length / 5
            elif self.close_profit > 25:
                    reward = self.close_profit / 5
            elif self.close_profit > 5:
                reward = 0
            else:
                reward = self.close_profit - 30
            self.close_profit = 0

        return obs, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.account_balance = INITIAL_ACCOUNT_BALANCE
        self.profit = 0
        self.trade_open_price = 0
        self.active_trade = 0
        self.close_profit = 0
        self.reward = 0
        self.profitable_buy = 0
        self.notprofitable_buy = 0
        self.profitable_sell = 0
        self.notprofitable_sell = 0
        self.pips_won = 0
        self.pips_lost = 0
        self.avg_length = [0]

        # Set the current step to a random point within the data frame
        self.current_step = 0

        return self._get_current_step_data()

    def render(self, mode='human', close=False):

        print(f'Step: {self.current_step}, active trade: {self.active_trade}, \
profit: {float(self.profit)}, acc balance: {float(self.account_balance)}, \
trade_open_price: {self.trade_open_price}, market level: {self.CurrentMarketLevel}, \
max_step: {self.max_step}')
