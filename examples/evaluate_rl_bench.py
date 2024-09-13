from functools import partial
import random

from absl import app, flags, logging
from envs.action_modes import DeltaEndEffectorPose
from envs.rl_bench_ur5_env import RLBenchUR5Env
import gymnasium as gym
import imageio
import jax
from matplotlib import pyplot as plt
import numpy as np
from rlbench.utils import name_to_task_class
import wandb.plot

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng
import wandb

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)
flags.DEFINE_integer(
    "checkpoint_step", None, "Step of the checkpoint used for evaluation."
)
flags.DEFINE_integer("action_horizon", 50, "Action horizon.")
flags.DEFINE_integer("rollouts", 3, "Number of evaluation rollouts.")
flags.DEFINE_enum(
    "task",
    "place_shape_in_shape_sorter",
    help="Type of a task.",
    enum_values=[
        "place_shape_in_shape_sorter",
        "pick_and_lift",
        "reach_target",
        "pick_and_lift_easy",
    ],
)
flags.DEFINE_integer(
    "variation",
    -1,
    help="Variation number. A value of -1 means that the variation is randomly sampled at each simulation reset. Variation numbers start from 0.",
)
flags.DEFINE_bool("headless", True, "Whether to start the simulator in headless mode.")
flags.DEFINE_bool(
    "record_video",
    False,
    "Whether to save the video of evaluation in 'outputs' directory.",
)
flags.DEFINE_integer(
    "steps", 200, "Number of maximum simulation steps in one evaluation run."
)
flags.DEFINE_enum(
    "robot_setup", "panda", help="Robot setup.", enum_values=["panda", "ur5"]
)
flags.DEFINE_enum("")


def convert_images_to_video(image_array, output_path, fps=30):
    """
    Converts a sequence of images (NumPy array) to an MP4 video using imageio.

    Parameters:
    - image_array: NumPy array of shape (num_frames, height, width, channels)
    - output_path: Path where the output video will be saved (e.g., 'output.mp4')
    - fps: Frames per second of the output video
    """
    # Initialize the video writer
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", format="mp4")

    # Write each frame to the video file
    for frame in image_array:
        writer.append_data(frame)

    # Close the writer to finalize the file
    writer.close()


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
    task_name = f"{FLAGS.task}-vision-v0"
    if use_proprio:
        task_name = f"{task_name}-proprio"

    gym.register(
        task_name,
        entry_point=lambda: RLBenchUR5Env(
            task_class=name_to_task_class(FLAGS.task),
            observation_mode="vision",
            render_mode="rgb_array",
            robot_setup=FLAGS.robot_setup,
            headless=FLAGS.headless,
            action_mode=DeltaEndEffectorPose(),
            proprio=use_proprio,
        ),
    )

    env = gym.make(task_name)

    if use_proprio:
        env = NormalizeProprio(env, model.dataset_statistics)

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
    for i in range(FLAGS.rollouts):
        obs, info = env.reset(options={"variation": FLAGS["variation"].value})

        # create task specification --> use model utility to create task dict with correct entries
        language_instructions = env.get_task()["language_instruction"]
        sampled_language_instruction = random.choice(language_instructions[0])
        task = model.create_tasks(texts=[sampled_language_instruction])

        images = [info["frame"]]
        episode_return = 0.0

        while len(images) < FLAGS.steps:
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
            actions = actions[0]
            actions_made.append(actions[: FLAGS.action_horizon])

            print(len(images), "Action", actions[: FLAGS.action_horizon])

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
                    np.array(images).transpose(0, 3, 1, 2)[::2], format="mp4", fps=15
                )
            }
        )

        if FLAGS.record_video:
            images_array = np.array(images)
            convert_images_to_video(
                images_array,
                f"outputs/{FLAGS.task}{i}_horizon_{FLAGS.action_horizon}.mp4",
                fps=30,
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
