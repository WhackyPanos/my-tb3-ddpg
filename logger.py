import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Logger():

    def __init__(self):
        self.columns = ['episode', 'x', 'y', 'theta', 'goal', 'state', 'action', 'reward']
        self.log_df = pd.DataFrame(columns = self.columns)

    def log(self, tuple):
        step = pd.DataFrame(tuple, index = self.columns)
        self.log_df = pd.concat([self.log_df, step.T], ignore_index=True)

    def save(self, path):
        self.log_df.to_csv(path)

    def read(self, path):
        self.log_df = pd.read_csv(path)
