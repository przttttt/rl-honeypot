import gymnasium as gym
from .response_env import ContextualResponseEnv, ResponseEnv
gym.register(id='rlhoneypotResponse-v0', entry_point='rlhoneypot.envs.response_env:ResponseEnv')
gym.register(id='rlhoneypotResponseContextual-v0', entry_point='rlhoneypot.envs.response_env:ContextualResponseEnv')
__all__ = ['ResponseEnv', 'ContextualResponseEnv']
