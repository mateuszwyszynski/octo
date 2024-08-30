"""
This script demonstrates how to load and rollout a finetuned Octo model.
We use the Octo model finetuned on ALOHA sim data from the examples/02_finetune_new_observation_action.py script.

For installing the ALOHA sim environment, clone: https://github.com/tonyzhaozh/act
Then run:
pip3 install opencv-python modern_robotics pyrealsense2 h5py_cache pyquaternion pyyaml rospkg pexpect mujoco==2.3.3 dm_control==1.0.9 einops packaging h5py

Finally, modify the `sys.path.append` statement below to add the ACT repo to your path.
If you are running this on a head-less server, start a virtual display:
    Xvfb :1 -screen 0 1024x768x16 &
    export DISPLAY=:1

To run this script, run:
    cd examples
    python3 03_eval_finetuned.py --finetuned_path=<path_to_finetuned_octo_checkpoint>
"""

from functools import partial
import random
import sys

from absl import app, flags, logging
import gymnasium as gym
import jax
from matplotlib import pyplot as plt
import numpy as np
from rlbench.utils import name_to_task_class
import wandb
import wandb.plot

from finetuning.envs.action_modes import UR5ActionMode
from finetuning.envs.rl_bench_env_adapter import RLBenchEnvAdapter  # noqa
from finetuning.envs.rl_bench_ur5_env import RLBenchUR5Env
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)
flags.DEFINE_integer("checkpoint_step", None, "Step of the checkpoint used for evaluation.")
flags.DEFINE_integer("action_horizon", 50, "Action horizon.")
flags.DEFINE_integer("rollouts", 3, "Number of evaluation rollouts.")
flags.DEFINE_enum(
    "task",
    "place_shape_in_shape_sorter",
    help="Type of a task.",
    enum_values=["place_shape_in_shape_sorter", "pick_and_lift"],
)
flags.DEFINE_integer(
    "variation",
    -1,
    help="Variation number. A value of -1 means that the variation is randomly sampled at each simulation reset. Variation numbers start from 0.",
)


def main(_):

    eval_config = {flag_name: FLAGS[flag_name].value for flag_name in FLAGS}

    # setup wandb for logging
    wandb.init(name="eval_rlbench", project="octo", config=eval_config)

    # load finetuned model
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(FLAGS.finetuned_path, step=FLAGS.checkpoint_step)

    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_primary": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_primary": ...
    #     }
    #   }
    ##################################################################################################################
    use_proprio = "proprio" in model.config["model"]["observation_tokenizers"]
    task_name = f"{FLAGS['task'].value}-vision-v0"
    if use_proprio:
        task_name = f"{task_name}-proprio"

    gym.register(
        task_name,
        entry_point=lambda: RLBenchEnvAdapter(
            RLBenchUR5Env(task_class=name_to_task_class("place_shape_in_shape_sorter"),
                    observation_mode='vision',
                    render_mode="rgb_array",
                    robot_setup="ur5",
                    headless=True,
                    action_mode=UR5ActionMode()),
            proprio=use_proprio
        ),
    )

    env = gym.make(task_name)

    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=FLAGS.action_horizon)

    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )

    # running rollouts
    actions_made = []
    for _ in range(FLAGS.rollouts):
        obs, info = env.reset(options={"variation": FLAGS["variation"].value})

        # create task specification --> use model utility to create task dict with correct entries
        language_instructions = env.get_task()["language_instruction"]
        sampled_language_instruction = random.choice(language_instructions[0])
        task = model.create_tasks(texts=[sampled_language_instruction])

        images = [info["frame"]]
        episode_return = 0.0

        while len(images) < 200:
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
            actions = actions[0]
            actions_made.append(actions[: FLAGS.action_horizon])

            print(len(images), "Action", actions)

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)
            images.extend([o for o in info["frame"]])
            episode_return += reward
            if done or trunc:
                break
        print(f"Episode return: {episode_return}")

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        wandb.log(
            {
                sampled_language_instruction: wandb.Video(
                    np.array(images).transpose(0, 3, 1, 2)[::2]
                )
            }
        )

    actions_made = np.concatenate(actions_made, axis=0)

    def vis_stats(vector, tag):
        assert len(vector.shape) == 2

        vector_mean = vector.mean(0)
        vector_std = vector.std(0)
        vector_min = vector.min(0)
        vector_max = vector.max(0)

        n_elems = vector.shape[1]
        fig = plt.figure(tag, figsize=(5 * n_elems, 10))
        for elem in range(n_elems):
            plt.subplot(1, n_elems, elem + 1)
            plt.hist(vector[:, elem], bins=20)
            plt.title(
                f"mean={vector_mean[elem]}\nstd={vector_std[elem]}\nmin={vector_min[elem]}\nmax={vector_max[elem]}",
            )

        wandb.log({tag: wandb.Image(fig)})

    vis_stats(actions_made, "action_stats")


if __name__ == "__main__":
    app.run(main)
