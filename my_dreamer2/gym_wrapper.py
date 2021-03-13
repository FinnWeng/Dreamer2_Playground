import gym
import cv2
import numpy as np


class Gym_Wrapper:
    def __init__(self, env, _isDiscrete, grayscale=True):
        self._env = env
        self._isDiscrete = _isDiscrete
        self._grayscale = grayscale
        self.crop_size = (160, 160)
        self.resize_size = (64, 64)
        self._actionRepeat = 4
        self._observation = []
        self.action_space = self._env.action_space
        self.action_dim = self._env.action_space.n
        self.observation_space = self._env.observation_space
        print("self.action_space.n:", self.action_space.n)
        print("self.observation_space:", self.observation_space)
        self.shape = self._env.observation_space.shape[:2] + (
            () if self._grayscale else (3,)
        )
        self._buffers = [
            np.empty(self.shape, dtype=np.uint8) for _ in range(2)
        ]  # to save observation to gray scale or others. Not replay buffer.

    def crop_ob(self, ob):
        h_start = self.shape[0] - self.crop_size[0]

        ob = ob[h_start:, :, :]
        return ob

    def reset(self):
        self._env.reset()
        if self._grayscale:
            self._env.ale.getScreenGrayscale(self._buffers[0])
        else:
            self._env.ale.getScreenRGB2(self._buffers[0])
        self._buffers[1].fill(0)

    def step(self, action):
        # for my version, I don't deal with repeat in gtm wrapper
        # ob, reward, done, info = self._env(action)

        ob, reward, done, _ = self._env.step(action)

        if self._grayscale:
            self._env.ale.getScreenGrayscale(self._buffers[0])
            # ob = ob[:, :, :1]
            ob = self._buffers[0]
            ob = ob[:, :, None] if self._grayscale else ob
            ob = self.crop_ob(ob)  # (160, 160, 1)
            ob = cv2.resize(ob, self.resize_size)  # (84, 84)
            ob = np.expand_dims(ob, axis=-1)  # (84, 84, 1)
        else:
            # self._env.ale.getScreenGrayscale(self._buffers[0])
            self._env.ale.getScreenRGB2(self._buffers[0])
            # ob = ob[:, :, :1]
            ob = self._buffers[0]
            ob = ob[:, :, None] if self._grayscale else ob
            ob = self.crop_ob(ob)  # (160, 160, 1)
            ob = cv2.resize(ob, self.resize_size)  # (84, 84)

        return ob, reward, done, _

    def render(self, mode="rgb_array"):
        return self._env.render(mode)


if __name__ == "__main__":
    task = "atari_Breakout"
    suite, task = task.split("_", 1)
    sticky_actions = True
    version = 0 if sticky_actions else 4

    name = "".join(word.title() for word in task.split("_"))

    # _env = gym.make("{}NoFrameskip-v{}".format(name, version))
    _env = gym.make("Breakout-v0")
    print(
        "_env:", _env.action_space
    )  # Discrete(4) <=> (-action_high, action_high)*action_dim

    print("_env: action_dim is None")
    print("_env:", _env.observation_space)  # Box(210, 160, 3), 0~255
    print("_env:", _env.render("rgb_array").shape)  # Box(210, 160, 3)

    wrapped_gym = Gym_Wrapper(_env, True)
    print("wrapped_gym.render():", wrapped_gym.render().shape)
    wrapped_gym.reset()
    for i in range(100):
        ob, reward, done, info = wrapped_gym.step([1])
        print("info:", info)
        print("ob:", ob.shape)
        cv2.imwrite("./train_gen_img/test_{}.jpg".format(i), ob)
