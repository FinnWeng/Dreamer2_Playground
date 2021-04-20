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
    
    return filename
    


def load_episodes(directory,limit = None):
    # rescan - output shape
    directory = pathlib.Path(directory).expanduser()
    
    cache = {}  # len 33
    # start_time = time.time()
    total_step = 0


    for filename in reversed(sorted(directory.glob("*.npz"))):
        

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
        cache[str(filename)] = episode
        total_step += (len(episode["rewards"]) - 1)
        if limit is not None:
            print("limit is not None")
            
            if total_step >= limit:
                print("over the limit!")
                break
    keys = list(cache.keys())  # which means each name of episode record in dir
    print("total_step:",total_step)
    print("keys:",len(keys))
    return cache
        


def sample_episodes(episodes,length=None, balance=False, seed=0):
    random = np.random.RandomState(seed)
    # for index in random.choice(len(keys), rescan):
    while True:
        episode = random.choice(list(episodes.values()))
        # episode = cache[keys[index]]
        if length:
            total = len(next(iter(episode.values())))
            # print("this sampled episode length is:",total)
            available = total - length
            

            if available < 1:
                print(f"Skipped short episode of length {available}.")
                continue
            if balance:
                index = min(random.randint(0, total), available)
            else:
                index = int(
                    random.randint(0, available+1)
                )  # randomly choose 0~available samples of  batch_length traj

            episode = {k: v[index : index + length] for k, v in episode.items()}
        
        yield episode



def load_dataset(episodes, config):  # load data from npz
    # episode = load_episodes(directory, 1)
    # episode = next(load_episodes(directory, 1))

    example = episodes[next(iter(episodes.keys()))]
    types = {k: v.dtype for k, v in example.items()}
    
    shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}
    print("shapes:",shapes)

    generator = lambda: sample_episodes(
        episodes, config.batch_length, config.oversample_ends,
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



class Optimizer(tf.Module):
    def __init__(self,name,lr,eps=1e-4, clip=None, wd, wd_pattern=r'.*',opt = "adam"):
        self._name = name
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            'adam': lambda: tf.optimizers.Adam(lr, epsilon=eps),
            'nadam': lambda: tf.optimizers.Nadam(lr, epsilon=eps),
            'adamax': lambda: tf.optimizers.Adamax(lr, epsilon=eps),
            'sgd': lambda: tf.optimizers.SGD(lr),
            'momentum': lambda: tf.optimizers.SGD(lr, 0.9),
        }[opt]()
    
    def __call__(self, tape, loss, modules):
        modules = modules if hasattr(modules, '__len__') else (modules,)
        varibs = tf.nest.flatten([module.variables for module in modules])
        grads = tape.gradient(loss, varibs)
        norm = tf.linalg.global_norm(grads)
        if self._clip:
            grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
        
        if self._wd:
            self._apply_weight_decay(varibs)
        
        self._opt.apply_gradients(zip(grads, varibs))
    
    def _apply_weight_decay(self, varibs):
        nontrivial = (self._wd_pattern != r'.*')
        if nontrivial:
        print('Applied weight decay to variables:')
        for var in varibs:
        if re.search(self._wd_pattern, self._name + '/' + var.name):
            if nontrivial:
            print('- ' + self._name + '/' + var.name)
            var.assign((1 - self._wd) * var)
