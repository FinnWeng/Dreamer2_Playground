# Dreamer2_Playground

This repo is my implementation which try to merge author's implementation into mine, in order to inspect detail of DreamerV2.

Author's implementation: https://github.com/danijar/dreamerv2 , it is more clear and well-organized, easy for reader to understand.

# result of training

The clear ascending trend of reward of an episode shows ability of DreamerV2. It takes one day, by one agent and one enviroment, to achieve this performance.

![alt text](https://github.com/FinnWeng/Dreamer2_Playground/blob/float16_latent/common/actor_reward.PNG "Actor Loss result")

Here shows an episode of Atari Breakout playing.

![alt text](https://github.com/FinnWeng/Dreamer2_Playground/blob/float16_latent/common/video.gif "DreamerV2 result")


# explain what I understand about DreamerV2

1. About imagine

The important of DreamerV1 is the concept that imagine world state in latent space. And DreamerV1 using VAE to make the posterior that combine world information close to prior without world information. The result distangles information into two group: information about state proceeding and information about the current timestep.

The DreamerV2 do one step forward: adapt beta-VAE to enhance ability of information distangle. In paper this trick is named KL balancing.

2. About target function

DreamerV1 adapt lambda-return to update actor. Lambda-Return may reduce target variance and bias.

For DreamerV2, instead of using only lambda-return, lambda-return is been combined with action log probability to create a form close to policy gradient to update actor. Value function also predicts lambda-return.

3. About discrete latent vatiable

The VAE of DreamV2 using one-hot distribution to make latent variable, and the one-hot distribution implementation contains direct gradient passing to pass gradient for update. The paper mentions that the reason why discrete latent variable works, is that it restrict representation ability of model.

# Something that I still not understand

1. Is mix-precision necessary?

2. In traing, the actor loss descending below zero. Then after 4M updating, it start to ascending. After about 8M update steps, the actor loss stucks around zero. Why it do not continue ascend?

