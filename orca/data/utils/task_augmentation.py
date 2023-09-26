"""
Contains basic logic for randomly zero-ing out keys in the task specification.
"""

from typing import Any, Dict, List, Tuple

import tensorflow as tf


def drop_keys_independent(
    traj: Dict[str, Any],
    drop_key_groups_probs: List[Tuple[List[str], float]],
    allow_drop_all: bool = False,
) -> Dict[str, Any]:
    """
    Independently drop keys in the tasks dictionary.

    :param traj: A dictionary containing trajectory data. should have a "tasks" key.
    :param drop_key_groups_probs: A list of tuples, where each tuple contains a list of keys and a dropout probability.
    :param allow_drop_all: If True, allow dropping all keys. Otherwise, if all keys are dropped, return the original
    :return: A dictionary with keys dropped out according to the specified probabilities.
    """

    # don't drop keys if there is no language instruction
    if tf.math.reduce_all(traj["tasks"]["language_instruction"] == ""):
        return traj

    tasks = traj["tasks"]
    new_tasks = tasks.copy()
    dropped_all = True

    for key_group, prob in drop_key_groups_probs:
        if not all(key in tasks for key in key_group):
            raise KeyError(
                f"keys {key_group} are not all present in tasks dictionary. tasks keys: {tasks.keys()}"
            )

        drop_group = tf.random.uniform([]) < prob
        dropped_all = dropped_all and drop_group
        for key in key_group:
            new_tasks[key] = tf.where(
                drop_group,
                tf.zeros_like(tasks[key])
                if tf.debugging.is_numeric_tensor(tasks[key])
                else "",
                tasks[key],
            )

    if not allow_drop_all and dropped_all:
        return traj

    traj["tasks"] = new_tasks
    return traj


def switch_keys(
    traj: Dict[str, Any],
    switch_key_groups_probs: List[Tuple[List[str], float]],
):
    """
    Randomly switch between keys in the tasks dictionary. Other keys are zeroed out.

    :param traj: A dictionary containing trajectory data. should have a "tasks" key.
    :param switch_key_groups_probs: A list of tuples, where each tuple contains a list of keys and their probability.
    :return: A dictionary with keys zeroed out according to the specified probabilities.
    """
    if tf.math.reduce_all(traj["tasks"]["language_instruction"] == ""):
        return traj

    tasks = traj["tasks"]
    new_tasks = tasks.copy()

    switch_probs = [prob for _, prob in switch_key_groups_probs]
    switch_group_idx = tf.random.categorical(tf.math.log([switch_probs]), 1)[0, 0]
    switch_key_groups_probs = switch_key_groups_probs.copy()
    switch_key_groups_probs.pop(int(switch_group_idx))

    for key_group, _ in switch_key_groups_probs:
        if not all(key in tasks for key in key_group):
            raise KeyError(
                f"keys {key_group} are not all present in tasks dictionary. tasks keys: {tasks.keys()}"
            )

        for key in key_group:
            new_tasks[keys] = (
                tf.zeros_like(tasks[key])
                if tf.debugging.is_numeric_tensor(tasks[key])
                else tf.fill(tf.shape(tasks[key]), "", tasks[key].dtype)
            )

    traj["tasks"] = new_tasks
    return traj
