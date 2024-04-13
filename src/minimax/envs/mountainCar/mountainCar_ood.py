from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import struct
import chex

from minimax.envs.registration import register

from .mountainCar import (
    MountainCar,
    EnvParams,
    EnvState
)

class MountainCarSingleton(MountainCar):
    def __init__(
        self,
        max_speed=0.07,
        goal_velocity=0.0,
        force=0.001,
        max_steps_in_episode=500,
        position=0,
        velocity=0,
        time=0,
        min_position=-1.2, 
        max_position=0.6,   
        gravity=0.0025,    
        complexity=1.0, 
        goal_position=0.5
    ):
        super().__init__(
            max_speed=max_speed,
            goal_velocity=goal_velocity,
            force=force,
            max_steps_in_episode=max_steps_in_episode
        )
        self.position = position
        self.velocity = velocity
        self.min_position = min_position
        self.max_position = max_position
        self.gravity = gravity
        self.complexity = complexity
        self.goal_position = goal_position

        self.params = EnvParams(
            max_speed=max_speed,
            goal_velocity=goal_velocity,
            force=force,
            max_steps_in_episode=max_steps_in_episode
        )

    def default_params(self):
        # Default environment parameters
        return EnvParams()


    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        state = EnvState(
                position = self.position,
                velocity = self.velocity,
                time = 0, 
                min_position = self.min_position,
                max_position = self.max_position,
                gravity = self.gravity, 
                complexity = self.complexity,
                goal_position = self.goal_position)
        return self.get_obs(state), state
    

class GoalNearMin(MountainCarSingleton):
    def __init__(
            self):
            super().__init__(
            position = 0.0,
            velocity = 0.0,
            min_position = -1.0,
            max_position = 2.8,
            gravity = 0.024,
            complexity = 1.0,
            goal_position = 0.8)


class GoalNearMax(MountainCarSingleton):
    def __init__(
            self):
            super().__init__(
            position = 0.0,
            velocity = 0.0,
            min_position = -1.0,
            max_position = 4.0,
            gravity = 0.024,
            complexity = 1.0,
            goal_position = 3.8)


class ToughCase(MountainCarSingleton):
    def __init__(
            self):
            super().__init__(
            position = 0.0,
            velocity = 0.0,
            min_position = -1.0,
            max_position = 9.0,
            gravity = 0.024,
            complexity = 2.0,
            goal_position = 7.8)

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(env_id='MountainCar-v0-GoalNearMin', entry_point=module_path + ':GoalNearMin')
register(env_id='MountainCar-v0-GoalNearMax', entry_point=module_path + ':GoalNearMax')
register(env_id='MountainCar-v0-ToughCase', entry_point=module_path + ':ToughCase')