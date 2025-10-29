import gymnasium as gym
import numpy as np

class RewardShaping(gym.Wrapper): 
    """
    Applies reward shaping bonuses for key intermediate achievements in Crafter.
    Bonuses are given only once per episode.
    """
    def __init__(self, env):
        super().__init__(env)
        self.bonuses_given = set()
        self.shaping_bonuses = {
            "collect_wood": 0.1,
            "place_table": 0.1,
            "collect_stone": 0.1,
            "make_wood_pickaxe": 0.1,
            "make_stone_pickaxe": 0.1,
            "place_furnace": 0.1,
            "collect_coal": 0.1,
        }
        self._internal_step = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.bonuses_given = set()
        self._internal_step = 0 
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._internal_step += 1

        shaped_reward = reward
        for ach_name, bonus_value in self.shaping_bonuses.items():
            full_ach_name = f"achievement_{ach_name}"
            if info.get(full_ach_name, 0) > 0 and ach_name not in self.bonuses_given:
                shaped_reward += bonus_value
                self.bonuses_given.add(ach_name)
        
        return observation, shaped_reward, terminated, truncated, info