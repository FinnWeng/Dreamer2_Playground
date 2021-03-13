import numpy as np
import utils
import uuid
import functools
import io
import pathlib
import datetime
import tensorflow as tf
import os
import time


def preprocess(episode_record, config):
    # episode_record = episode_record.copy()
    with tf.device("cpu:0"):
        episode_record["obs"] = tf.cast(episode_record["obs"], tf.float32) / 255.0 - 0.5
        episode_record["obp1s"] = (
            tf.cast(episode_record["obp1s"], tf.float32) / 255.0 - 0.5
        )
        # clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[
        #     config.clip_rewards
        # ]  # default none
        # episode_record["rewards"] = clip_rewards(episode_record["rewards"])
    return episode_record


def reverse_presprocess(episode_record):
    # episode_record = episode_record.copy()
    with tf.device("cpu:0"):
        episode_record["obs"] = tf.cast((episode_record["obs"] + 0.5) * 255.0, tf.int32)
        episode_record["obp1s"] = tf.cast(
            (episode_record["obp1s"] + 0.5) * 255.0, tf.int32
        )
        # clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[
        #     config.clip_rewards
        # ]  # default none
        # episode_record["rewards"] = clip_rewards(episode_record["rewards"])
    return episode_record


def save_episode(directory, episode_record):

    # episode_record = {
    #     k: [t[k] for t in episode_record] for k in episode_record.keys()
    # }  # list of dict to  {k: list of value}

    # print("episode_record['ob']:", episode_record["ob"].shape)

    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    identifier = str(uuid.uuid4().hex)
    length = len(
        episode_record["rewards"]
    )  # the total reward for 1 step, 4 same actions
    filename = directory / f"{timestamp}-{identifier}-{length}.npz"
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode_record)
        f1.seek(0)
        with filename.open("wb") as f2:
            f2.write(f1.read())


def load_episodes(directory, rescan, length=None, balance=False, seed=0):
    # rescan - output shape
    directory = pathlib.Path(directory).expanduser()
    random = np.random.RandomState(seed)
    cache = {}  # len 33
    # start_time = time.time()
    while True:
        for filename in directory.glob("*.npz"):
            # print("filename:", filename)
            if filename not in cache:
                try:
                    with filename.open("rb") as f:
                        # print("start loading!")
                        episode = np.load(f)
                        # episode = np.load(f,allow_pickle=True)

                        # print("finish loading!")

                        episode = {
                            k: episode[k] for k in episode.keys()
                        }  # dict_keys(['image', 'action', 'reward', 'discount'])

                except Exception as e:
                    print(f"Could not load episode: {e}")
                    continue
                cache[filename] = episode
        keys = list(cache.keys())  # which means each name of episode record in dir
        # print("keys:", keys)
        # filename_gen = directory.glob("*.npz")
        # filename_list = list(filename_gen)
        # for index in random.choice(len(filename_list), rescan):
        #     filename = filename_list[index]
        #     try:
        #         with filename.open("rb") as f:
        #             # print("start loading!")
        #             # print("filename:",f)
        #             episode = np.load(f)
        #             # episode = np.load(f,allow_pickle=True)

        #             # print("finish loading!")

        #             episode = {
        #                 k: episode[k] for k in episode.keys()
        #             }  # dict_keys(['image', 'action', 'reward', 'discount'])

        #     except Exception as e:
        #         print(f"Could not load episode: {e}")
        #         continue
        #     cache[filename] = episode
        # keys = list(cache.keys())

        # end_time = time.time()
        # print("loadtime",end_time - start_time)

        # for index in random.choice(len(keys), rescan):
        #     episode = cache[keys[index]]  # {k: list of value}
        #     if length:
        #         total = len(next(iter(episode.values())))

        #         available = total - length

        #         if available < 1:
        #             print(f"Skipped short episode of length {available}.")
        #             continue
        #         if balance:
        #             index = min(random.randint(0, total), available)
        #         else:
        #             index = int(
        #                 random.randint(0, available)
        #             )  # randomly choose 0~available samples of  batch_length traj

        #         episode = {k: v[index : index + length] for k, v in episode.items()}

        #     yield episode

        # for not using generater
        episode_list = []
        for index in random.choice(len(keys), rescan):
            episode = cache[keys[index]]  # {k: list of value}
            if length:
                total = len(next(iter(episode.values())))

                available = total - length

                if available < 1:
                    print(f"Skipped short episode of length {available}.")
                    continue
                if balance:
                    index = min(random.randint(0, total), available)
                else:
                    index = int(
                        random.randint(0, available)
                    )  # randomly choose 0~available samples of  batch_length traj

                episode = {k: v[index : index + length] for k, v in episode.items()}

            # # # turn into list of dict
            # print("rescan:", rescan)  # 100
            # print("length:", length)  # 10
            # print("episode['obs'].shape():", episode["obs"].shape)  # (10, 64, 64, 3)
            episode_list.append(episode)
        episode = {
            k: np.stack([episode[k] for episode in episode_list], 0)
            for k, v in episode_list[0].items()
        }
        return episode


def load_dataset(directory, config):  # load data from npz
    episode = load_episodes(directory, 1)
    # episode = next(load_episodes(directory, 1))
    types = {k: v.dtype for k, v in episode.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}

    generator = lambda: load_episodes(
        directory, config.train_steps, config.batch_length, config.dataset_balance
    )
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)

    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(preprocess, config=config))
    dataset = dataset.prefetch(10)
    return dataset


class slices_dataset_generator:
    def __init__(self, directory, config):
        self.directory = directory
        self.config = config
        self.dataset = self.reload(self.directory, self.config)

    def reload(self, directory, config):

        dataset_slice = load_episodes(
            directory, config.train_steps, config.batch_length, config.dataset_balance
        )
        dataset = tf.data.Dataset.from_tensor_slices(dataset_slice)

        dataset = dataset.batch(config.batch_size, drop_remainder=True)
        dataset = dataset.map(functools.partial(preprocess, config=config))
        dataset = dataset.prefetch(10)
        return dataset.as_numpy_iterator()

    def __call__(self):
        while True:
            try:
                return next(self.dataset)

            except (StopIteration, tf.errors.OutOfRangeError):
                self.dataset = self.reload(self.directory, self.config)
                print("reload dataset until success!")
                # return next(self.dataset)
