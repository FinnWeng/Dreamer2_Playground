import gym
import cv2
import numpy as np


class Gym_Wrapper:
    def __init__(
        self,
        name,
        action_repeat=4,
        size=(84, 84),
        grayscale=True,
        noops=30,
        life_done=False,
        sticky_actions=True,
        all_actions=False,
    ):

        self.crop_size = (160, 160)
        self.resize_size = (64, 64)
        self._actionRepeat = action_repeat
        self._observation = []
        # self.action_space = self._env.action_space
        import gym.wrappers
        import gym.envs.atari

        env = gym.envs.atari.AtariEnv(
            game=name,
            obs_type="image",
            frameskip=1,
            repeat_action_probability=0.25 if sticky_actions else 0.0,
            full_action_space=all_actions,
        )
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec("NoFrameskip-v0")
        env = gym.wrappers.AtariPreprocessing(
            env, noops, self._actionRepeat, size[0], life_done, grayscale
        )
        self._env = env
        self._grayscale = grayscale

        self.action_dim = self._env.action_space.n
        self.observation_space = self._env.observation_space
        # print("self.action_space.n:", self.action_space.n)
        # print("self.observation_space:", self.observation_space)
        self.shape = self._env.observation_space.shape[:2] + (
            () if self._grayscale else (3,)
        )
    
    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.discrete = True
        return space

    def reset(self):

        image = self._env.reset()
        # print("image:",image.shape)
        if self._grayscale:
            image = image[..., None]
        return image

    def step(self, action):
        # for my version, I don't deal with repeat in gtm wrapper'
        # ob, reward, done, info = self._env(action)

        ob, reward, done, _ = self._env.step(action)

        if self._grayscale:
            ob = ob[..., None]

        return ob, reward, done, _

    # def render(self, mode="rgb_array"):
    #     return self._env.render(mode)

    def render(self, mode="rgb_array"):
        return self._env.render()


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
