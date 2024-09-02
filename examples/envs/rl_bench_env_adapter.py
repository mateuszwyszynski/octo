import copy
from typing import List

import dlimp as dl
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from PIL import Image

from ...examples.envs.rl_bench_ur5_env import RLBenchUR5Env


# TODO: This class made sense, when we used RLBenchEnv. While we are using our own RLBenchUR5Env,
# the RLBenchEnvAdapter class should be removed and all environment logic should be moved to RLBenchUR5Env.
class RLBenchEnvAdapter(gym.Env):
    def __init__(
        self,
        rl_bench_env: RLBenchUR5Env,
        im_size: int = 256,
        proprio: bool = True,
        seed: int = 1234,
    ):
        self._env = rl_bench_env
        self.proprio = proprio

        observation_space_dict = {
            **{
                f"image_{i}": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                )
                for i in ["primary", "wrist"]
            },
        }
        if proprio:
            observation_space_dict["proprio"] = gym.spaces.Box(
                low=-np.infty * np.ones(7),
                high=np.infty * np.ones(7),
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_space_dict)
        self.action_space = self._env.action_space
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        observation, reward, terminated, _, _ = self._env.step(action)
        obs = self._extract_obs(observation)

        # It assumes that reward == 1.0 means success
        if reward == 1.0:
            self._episode_is_success = 1

        rendered_frame = self.render(obs)

        return obs, reward, terminated, False, {"frame": rendered_frame}

    def render(self, obs=None):
        rendered_frame = self._env.render()

        if rendered_frame is not None:

            wrist_img = Image.fromarray(obs["image_wrist"].numpy())
            wrist_img = wrist_img.resize((360, 360), Image.Resampling.BILINEAR)
            resized_wrist_img = np.array(wrist_img)

            return np.concatenate([rendered_frame, resized_wrist_img], axis=1)

    def reset(self, **kwargs):
        if kwargs["options"]["variation"] == -1:
            self._env.rlbench_task_env.sample_variation()
        else:
            self._env.rlbench_task_env.set_variation(kwargs["options"]["variation"])

        obs, info = self._env.reset(**kwargs)

        obs = self._extract_obs(obs)
        self._episode_is_success = 0
        self.language_instruction = info["text_descriptions"]

        rendered_frame = self.render(obs)

        return obs, {"frame": rendered_frame}

    def _extract_obs(self, obs):
        curr_obs = {
            "image_primary": obs["front_rgb"],
            "image_wrist": obs["wrist_rgb"],
        }

        if self.proprio:
            curr_obs["proprio"] = np.concatenate(
                [obs["joint_positions"], obs["gripper_open"]]
            )

        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        return curr_obs

    def get_task(self):
        return {
            "language_instruction": [self.language_instruction],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }
