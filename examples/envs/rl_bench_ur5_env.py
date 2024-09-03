import dlimp as dl
from PIL import Image
from typing import Union

import numpy as np
from gymnasium import spaces
import gymnasium as gym
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode


from rlbench.action_modes.action_mode import JointPositionActionMode
from rlbench.environment import Environment
from rlbench.gym import RLBenchEnv
from rlbench.observation_config import ObservationConfig


class RLBenchUR5Env(RLBenchEnv):
    """An gym wrapper for RLBench."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(
            self, task_class, observation_mode='state',
            render_mode: Union[None, str] = None, action_mode=None,
            robot_setup: str="panda", headless: bool=True,
            im_size: int = 256, proprio: bool = True, seed: int = 1234,
            ):
        self.task_class = task_class
        self.observation_mode = observation_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        obs_config = ObservationConfig()
        obs_config.front_camera.image_size = (256, 256)
        obs_config.wrist_camera.image_size = (256, 256)
        if observation_mode == 'state':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == 'vision':
            obs_config.set_all(True)
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)

        if robot_setup == "ur5":
            obs_config.gripper_touch_forces = False

        self.obs_config = obs_config
        if action_mode is None:
            action_mode = JointPositionActionMode()
        self.action_mode = action_mode

        self.rlbench_env = Environment(
            action_mode=self.action_mode,
            obs_config=self.obs_config,
            headless=headless,
            robot_setup=robot_setup
        )
        self.rlbench_env.launch()
        self.rlbench_task_env = self.rlbench_env.get_task(self.task_class)
        if render_mode is not None:
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            self.gym_cam = VisionSensor.create([640, 360])
            self.gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == "human":
                self.gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self.gym_cam.set_render_mode(RenderMode.OPENGL3)
        _, obs = self.rlbench_task_env.reset()

        gym_obs = self._extract_obs(obs)
        self.observation_space = {}
        for key, value in gym_obs.items():
            if "rgb" in key:
                self.observation_space[key] = spaces.Box(
                    low=0, high=255, shape=value.shape, dtype=value.dtype)
            else:
                self.observation_space[key] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=value.shape, dtype=value.dtype)
        self.observation_space = spaces.Dict(self.observation_space)

        action_low, action_high = action_mode.action_bounds()
        self.action_space = spaces.Box(
            low=np.float32(action_low), high=np.float32(action_high), shape=self.rlbench_env.action_shape, dtype=np.float32)

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
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)


    def _extract_obs(self, rlbench_obs):
        gym_obs = {} 
        for state_name in ["joint_velocities", "joint_positions", "joint_forces", "gripper_open", "gripper_pose", "gripper_joint_positions", "gripper_touch_forces", "task_low_dim_state"]:
            state_data = getattr(rlbench_obs, state_name)
            if state_data is not None:
                state_data = np.float32(state_data)
                if np.isscalar(state_data):
                    state_data = np.asarray([state_data])
                gym_obs[state_name] = state_data
                
        if self.observation_mode == 'vision':
            gym_obs.update({
                "left_shoulder_rgb": rlbench_obs.left_shoulder_rgb,
                "right_shoulder_rgb": rlbench_obs.right_shoulder_rgb,
                "wrist_rgb": rlbench_obs.wrist_rgb,
                "front_rgb": rlbench_obs.front_rgb,
            })
        return gym_obs


    def step(self, action):
        observation, reward, terminated, _, _ = super().step(action)
        obs = self.extract_obs(observation)

        # It assumes that reward == 1.0 means success
        if reward == 1.0:
            self._episode_is_success = 1

        rendered_frame = self.render(obs)

        return obs, reward, terminated, False, {"frame": rendered_frame}


    def render(self, obs=None):
        rendered_frame = super().render()

        if rendered_frame is not None:

            wrist_img = Image.fromarray(obs["image_wrist"].numpy())
            wrist_img = wrist_img.resize((360, 360), Image.Resampling.BILINEAR)
            resized_wrist_img = np.array(wrist_img)

            return np.concatenate([rendered_frame, resized_wrist_img], axis=1)


    def reset(self, **kwargs):
        if kwargs["options"]["variation"] == -1:
            self.rlbench_task_env.sample_variation()
        else:
            self.rlbench_task_env.set_variation(kwargs["options"]["variation"])

        obs, info = super().reset(**kwargs)

        obs = self.extract_obs(obs)
        self._episode_is_success = 0
        self.language_instruction = info["text_descriptions"]

        rendered_frame = self.render(obs)

        return obs, {"frame": rendered_frame}


    def extract_obs(self, obs):
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
