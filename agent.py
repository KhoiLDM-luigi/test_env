from rl.agents import DQNAgent
import numpy as np

class ActionMask:
    def __init__(self, num_action) -> None:
        self.num_action = num_action
        self._invalid_action = []
        
    def update_invalid_action(self, actions: list):
        for i in actions:
            if i >= self.num_action or i < 0: 
                raise Exception(f"Invalid action {i}")
        self._invalid_action = actions

    def mask(self, softmax):
        assert len(softmax) == self.num_action, 'invalid softmax space'
        for i in self._invalid_action:
            softmax[i] = -np.inf
        return softmax

class CustomDQNAgent(DQNAgent):
    def __init__(self, *args, action_mask: ActionMask, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_mask = action_mask

    def compute_q_values(self, state):
        q_values = super().compute_q_values(state)
        return self.action_mask.mask(q_values)

