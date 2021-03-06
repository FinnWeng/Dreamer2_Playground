import os, inspect
import dreamer_play
from gym_wrapper import Gym_Wrapper
import gym
from net.dreamer_net import Dreamer
import functools
import pathlib
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

# import warnings
# warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
# warnings.filterwarnings("ignore", ".*TensorFloat-32 matmul/conv*")
# tf.get_logger().setLevel("ERROR")


class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def define_config():
    config = AttrDict()
    # General.
    config.logdir = pathlib.Path("./episode_log")
    config.seed = 0
    config.steps = 5e6
    config.eval_every = 1e4
    config.log_every = 1e3

    config.gpu_growth = True
    config.precision = 16
    # Environment.
    config.task = "breakout"
    # config.task = 'atari_SpaceInvaders'
    config.size = [64, 64]
    config.is_discrete = True
    config.grayscale = True
    config.envs = 1
    config.parallel = "none"
    config.action_repeat = 4
    config.time_limit = 108000
    config.prefill = 50000
    config.eval_noise = 0.0
    config.clip_rewards = "tanh"
    # Model.
    config.dyn_cell = "gru_layer_norm"
    config.units = 400
    config.dyn_stoch = 32
    config.dyn_deter = 600
    config.dyn_hidden = 600
    config.dyn_discrete = 32
    config.dense_act = "elu"
    config.cnn_act = "elu"
    config.cnn_depth = 48
    config.encoder_kernels = [4, 4, 4, 4]
    config.decoder_kernels = [5, 5, 6, 6]

    config.pcont = True
    config.pcont_scale = 5.0
    config.eta_x = 1
    config.eta_r = 1

    config.weight_decay = 1e-6

    # Training.
    config.batch_size = 50

    config.batch_length = 50
    # config.batch_length = 10
    config.train_every = 16

    config.max_dataset_steps = 2e6
    config.oversample_ends = True
    config.slow_value_target = True
    config.slow_actor_target = True
    config.slow_target_update = 100
    config.slow_target_fraction = 1
    config.opt = "adam"

    config.pretrain = 0
    config.model_lr = 2e-4
    config.actor_lr = 4e-5
    config.value_lr = 1e-4
    config.grad_clip = 100.0
    config.actor_grad_clip = 100.0
    config.value_grad_clip = 100.0
    config.opt_eps = 1e-5
    config.kl_balance = "0.8"
    config.kl_scale = 0.1
    config.kl_free = 0.0
    config.kl_forward = False
    # Behavior.
    config.discount = 0.999
    config.imag_gradient = "both"
    config.imag_gradient_mix = "linear(0.1,0,2.5e6)"
    config.discount_lambda = 0.95
    config.horizon = 15
    #   config.action_dist = 'tanh_normal' # for continous action
    config.action_dist = "onehot"  # for onehot action
    config.actor_entropy = "linear(3e-3,3e-4,2.5e6)"
    config.actor_state_entropy = 0.0

    config.action_init_std = 1.0
    config.expl = "epsilon_greedy"
    config.expl_amount = 0.0
    config.behavior_stop_grad = True
    config.future_entropy = False

    return config


if __name__ == "__main__":

    config = define_config()
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    # self._dataset = iter(self.load_dataset(self.datadir, self._c))
    gpus = tf.config.experimental.list_physical_devices("GPU")

    tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_policy(prec.Policy("mixed_float16"))

    mode = "train"

    train_wrapped_gym = Gym_Wrapper(
        config.task,
        config.action_repeat,
        config.size,
        grayscale=config.grayscale,
        life_done=False and (mode == "train"),
        sticky_actions=True,
        all_actions=True,
    )

    play_process = dreamer_play.Play(train_wrapped_gym, Dreamer, config, mode)

    while True:
        """
        for longest play, play_process.play_records will be 625. Since 625 is map play step 10000/(batch_size*TD_size)
        for single play record, since I adopt act repeat and TD method, so it contain repeat_time*TD_size frames
        """
        print("play_process.play_records:", len(play_process.play_records))
        play_process.collect(must_be_whole_episode=False, prefill=False)
        print("play_process.play_records:", len(play_process.play_records))
        # mean_reward = play_process.dreaming_update()
        # print("rewards:", mean_reward)
