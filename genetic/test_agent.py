from pommerman.agents.base_agent import BaseAgent
from gym.spaces import Discrete

class TestAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs: dict, action_space: Discrete):
        return action_space.sample()
