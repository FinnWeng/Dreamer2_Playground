import os, inspect
import dreamer_play
from gym_wrapper import Gym_Wrapper
import gym
from net.dreamer_net import Dreamer
import functools
import pathlib


class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def define_config():
    config = AttrDict()
    # General.
    #   config.logdir = pathlib.Path('.')
    config.logdir = pathlib.Path("./episode_log")
    config.seed = 0
    config.steps = 5e6
    config.eval_every = 1e4
    config.log_every = 1e3
    config.log_scalars = True
    config.log_images = True
    config.gpu_growth = True
    config.precision = 16
    # Environment.
    #   config.task = 'dmc_walker_walk'
    config.task = "atari_Breakout"
    #   config.task = 'atari_SpaceInvaders'
    config.is_discrete = True
    config.grayscale = False
    config.envs = 1
    config.parallel = "none"
    config.action_repeat = 2
    config.time_limit = 1000
    config.prefill = 5000
    config.eval_noise = 0.0
    config.clip_rewards = "none"
    # Model.
    config.deter_size = 200
    config.stoch_size = 30
    config.num_units = 400
    config.dense_act = "elu"
    config.cnn_act = "relu"
    config.cnn_depth = 32
    #   config.pcont = False
    config.pcont = True
    config.free_nats = 3.0
    config.kl_scale = 1.0
    config.pcont_scale = 10.0
    config.weight_decay = 0.0
    config.weight_decay_pattern = r".*"
    # Training.
    config.batch_size = 50

    # config.batch_length = 50
    config.batch_length = 10
    config.train_every = 1000
    config.train_steps = 100
    config.pretrain = 100
    config.model_lr = 6e-4
    config.value_lr = 8e-5
    config.actor_lr = 8e-5
    config.grad_clip = 100.0
    config.dataset_balance = False
    # Behavior.
    config.discount = 0.99
    config.disclam = 0.95
    config.horizon = 15
    #   config.action_dist = 'tanh_normal' # for continous action
    config.action_dist = "onehot"  # for onehot action
    config.action_init_std = 5.0
    #   config.expl = 'additive_gaussian'
    config.expl = "epsilon_greedy"
    config.expl_amount = 0.3
    config.expl_decay = 0.0
    config.expl_min = 0.0
    return config


if __name__ == "__main__":

    task = "atari_Breakout"
    suite, task = task.split("_", 1)
    sticky_actions = True
    version = 0 if sticky_actions else 4

    name = "".join(word.title() for word in task.split("_"))

    _env = gym.make("{}NoFrameskip-v{}".format(name, version))

    config = define_config()

    wrapped_gym = Gym_Wrapper(_env, True, config.grayscale)

    config_dreamer = functools.partial(Dreamer, config=config)

    play_process = dreamer_play.Play(wrapped_gym, Dreamer, config)

    while True:
        """
        for longest play, play_process.play_records will be 625. Since 625 is map play step 10000/(batch_size*TD_size)
        for single play record, since I adopt act repeat and TD method, so it contain repeat_time*TD_size frames
        """
        print("play_process.play_records:", len(play_process.play_records))
        play_process.collect(using_random_policy=False, must_be_whole_episode=False)
        print("play_process.play_records:", len(play_process.play_records))
        mean_reward = play_process.dreaming_update()
        print("rewards:", mean_reward)