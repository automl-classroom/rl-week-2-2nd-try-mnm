from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int | None = None):
        """Initializes the observation and action space for the environment."""
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.P = self.get_transition_matrix()
        self.horizon = 10
        self.current_steps = 0
        self.position = 0
        self.rewards = np.array([0, 1])  # Rewards for actions 0 and 1
        self.states = np.array([0, 1])  # Possible states

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment to the initial state."""
        self.current_steps = 0
        self.position = 0
        return self.position, {}

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            Action to take (0: left, 1: right).

        Returns
        -------
        next_state : int
            The resulting position of the rover.
        reward : float
            The reward at the new position.
        terminated : bool
            Whether the episode ended due to task success (always False here).
        truncated : bool
            Whether the episode ended due to reaching the time limit.
        info : dict
            An empty dictionary.
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        p = float(self.P[self.position, action, action])
        follow = self.rng.random() < p
        a_used = action if follow else 1 - action

        delta = -1 if a_used == 0 else 1
        self.position = int(max(0, min(self.states[-1], self.position + delta)))

        reward = float(self.rewards[self.position])
        terminated = False
        truncated = self.current_steps >= self.horizon

        return self.position, reward, terminated, truncated, {}

    def get_reward_per_action(self) -> np.ndarray:
        """Returns the reward per action for each state."""
        nS, nA = self.observation_space.n, self.action_space.n
        R = np.zeros((nS, nA), dtype=float)
        for s in range(nS):
            for a in range(nA):
                R[s, a] = a  # reward is equal to the action taken
        return R

    def get_transition_matrix(self):
        """Returns the transition matrix of the environment."""
        return np.array(
            [
                [
                    [1, 0],  # action 0 from state 0
                    [0, 1],
                ],  # action 1 from state 0
                [
                    [1, 0],  # action 0 from state 1
                    [0, 1],
                ],
            ]
        )  # action 1 from state 1


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        pass
