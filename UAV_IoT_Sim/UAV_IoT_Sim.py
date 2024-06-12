# Dependecies to Import
import pandas as pd

# Simulation Files to Import
from UAV_IoT_Sim import Environment, IoT_Device


class make_env:
    def __init__(
        self,
        scene: str = "test",
        num_sensors: int = 50,
        num_ch: int = 5,
        max_num_steps: int = 720
    ):
        self.scene = scene
        self._env = Environment.sim_env(scene, num_sensors, num_ch, max_num_steps)
        self._num_sensors = num_sensors
        self.num_ch = num_ch
        self._max_steps = max_num_steps
        
        self.curr_step = 0
        self.curr_state = [[0, 0, 0] for _ in range(self.num_ch + 6)]
        self.last_action = 0

        self.archived_state = [[0, 0, 0] for _ in range(self.num_ch + 6)]
        self.archived_action = 0

        self.curr_reward = 0
        self.curr_info = {
            "Last_Action": None,
            "Reward_Change": 0.0,          # -> Change in reward at step
            "Avg_Age": 0.0,               # -> avgAoI
            "Peak_Age": 0.0,              # -> peakAoI
            "Data_Distribution": 0.0,     # -> Distribution of Data
            "Total_Data_Change": 0.0,      # -> Change in Total Data
            "Total_Data": 0.0,            # -> Total Data
            "Crashed": False,
            "Truncated": False
        }

        self._aoi_threshold = 60
        self.truncated = False
        self.terminated = False
        self._curr_total_data = 0
    
    def reset(self):
        if self.scene == "test":
            for sensor in range(self._num_sensors):
                self._env.sensorTable.iloc[sensor,0].reset()
            for CH in range(self.num_ch):
                self._env.CHTable.iloc[CH, 0].reset()
            self._env.initInterference()
            
            self.curr_step = 0
            self.curr_state = [[0, 0, 0] for _ in range(self.num_ch + 6)]
            self.last_action = 0

            self.archived_state = [[0, 0, 0] for _ in range(self.num_ch + 6)]
            self.archived_action = 0

            self.curr_reward = 0
            self.curr_info = {
                "Last_Action": None,
                "Reward_Change": 0.0,          # -> Change in reward at step
                "Avg_Age": 0.0,               # -> avgAoI
                "Peak_Age": 0.0,              # -> peakAoI
                "Data_Distribution": 0.0,     # -> Distribution of Data
                "Total_Data_Change": 0.0,      # -> Change in Total Data
                "Total_Data": 0.0,            # -> Total Data
                "Crashed": False, 
                "Truncated": False
            }

            self.truncated = False
            self.terminated = False
            self._curr_total_data = 0
            return self._env
    
    def step(self):
        old_state = [[0, 0, 0] for _ in range(self.num_ch + 6)]

        if not self.terminated:
            if self.curr_step < self._max_steps:
                x = self.curr_step/60 + 2
                alpha = 1.041834 - 0.6540587 * x + 0.4669073 * pow(x, 2) - 0.1225805 * pow(x, 3) + 0.0137882 * pow(x, 4) - 0.0005703625 * pow(x, 5)
            	
                for sens in range(self._num_sensors):
                    self._env.sensorTable.iloc[sens, 0].harvest_energy(alpha, self._env, self.curr_step)
                    self._env.sensorTable.iloc[sens, 0].harvest_data(self.curr_step)

                for CH in range(self.num_ch):
                    old_state[CH][0] = CH
                    ch_energy = self._env.CHTable.iloc[CH, 0].harvest_energy(alpha, self._env, self.curr_step)
                    old_state[CH][1] = ch_energy
                    ch_data = self._env.CHTable.iloc[CH, 0].ch_download(self.curr_step)
                    old_state[CH][2] = ch_data

                self.curr_step += 1
            else:
                self.truncated = True
                self.curr_info = {
                    "Last_Action": self.last_action,
                    "Reward_Change": 0,        # -> Change in reward at step
                    "Avg_Age": 0,                   # -> avgAoI
                    "Peak_Age": 0,                 # -> peakAoI
                    "Data_Distribution": 0,     # -> Distribution of Data
                    "Total_Data_Change": 0,      # -> Change in Total Data
                    "Total_Data": self._curr_total_data, # -> Total Data
                    "Crashed": self.terminated,         # -> True if UAV is crashed
                    "Truncated": self.truncated         # -> Max episode steps reached
                }
        else:
            self.curr_reward = 0
            self.curr_info = {
                "Last_Action": self.last_action,
                "Reward_Change": 0,        # -> Change in reward at step
                "Avg_Age": 0,                   # -> avgAoI
                "Peak_Age": 0,                 # -> peakAoI
                "Data_Distribution": 0,     # -> Distribution of Data
                "Total_Data_Change": 0,      # -> Change in Total Data
                "Total_Data": self._curr_total_data, # -> Total Data
                "Crashed": self.terminated,         # -> True if UAV is crashed
                "Truncated": self.truncated         # -> Max episode steps reached
            }

        return old_state
