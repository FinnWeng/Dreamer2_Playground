import tensorflow as tf
import numpy as np
import time
import cv2

import pickle
import utils
import tools
from tensorflow.keras.mixed_precision import experimental as prec
import copy


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


class Play:
    def __init__(self, env, model_maker, config, training=True):
        self.env = env

        # self.act_space = self.env.action_space

        self._c = config
        self._precision = config.precision
        self._float = prec.global_policy().compute_dtype

        # self.ob, _, _, _ = self.env.step(
        #     self.env._env.action_space.sample()
        # )  # whether it is discrete or not, 0 is proper
        self.ob = self.env.reset()
        acts = self.env.action_space
        self.random_actor = tools.OneHotDist(tf.zeros_like(acts.low)[None])

        self._c.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        print("self._c.num_actions:", self._c.num_actions)
        # self.batch_size = 16
        self.batch_size = self._c.batch_size
        self.batch_length = (
            self._c.batch_length
        )  # when it is not model-based learning, consider it controling the replay buffer
        self.TD_size = 1  # no TD
        self.play_records = []

        self.advantage = True

        self.total_step = 1
        self.save_play_img = False
        self.RGB_array_list = []
        self.episode_reward = 0
        self.episode_step = 1  # to avoid devide by zero
        self.datadir = self._c.logdir / "episodes"

        self._writer = tf.summary.create_file_writer(
            "./tf_log", max_queue=1000, flush_millis=20000
        )

        if training:
            self.prefill_and_make_dataset()
        else:
            pass

        with tf.device("cpu:1"):
            self._step = tf.Variable(count_steps(self.datadir), dtype=tf.int64)

        self._c.actor_entropy = lambda x=self._c.actor_entropy: tools.schedule(
            x, self._step
        )
        self._c.actor_state_entropy = (
            lambda x=self._c.actor_state_entropy: tools.schedule(x, self._step)
        )
        self._c.imag_gradient_mix = lambda x=self._c.imag_gradient_mix: tools.schedule(
            x, self._step
        )
        self.model = model_maker(self.env, training, self._step, self._writer, self._c)

    def random_policy(self):
        action = self.random_actor.sample()
        return action

    def prefill_and_make_dataset(self):
        # since it casuse error when random choice zero object in self.load_dataset
        self.episodes = utils.load_episodes(
            self.datadir, limit=self._c.max_dataset_steps
        )
        while True:
            self.collect(must_be_whole_episode=True, prefill=True)
            if self.total_step >= self._c.prefill:
                self.reset_total_step()
                break
        self._dataset = iter(utils.load_dataset(self.episodes, self._c))

        print("inspect initial episode!!!!")

        # for i in range(500):

        #     data =  self._dataset()
        #     for j in range(50):
        #         k = 5
        #         # for k in range(10):
        #         obs = (data["obs"][j,k,:,:,:]+0.5)*255
        #         print(i,j,k)

        #         # print('data["obs"].shape:',obs.shape) # (50, 10, 64, 64, 3)
        #         cv2.imshow("obs", obs)
        #         cv2.waitKey(1)

        #     # cv2.destroyAllWindows()

    def act_repeat(self, env, act, rendering=False):
        ob, reward, done, info = self.env.step(act)
        if rendering:
            rgb_array = self.env.render()

        return ob, reward, done, info

    def TD_dict_to_TD_train_data(self, batch_data, advantage=False):
        # list of dict of batch_size,TD_size,{4}
        obs = []
        acts = []
        obp1s = []
        rewards = []
        dones = []
        discounts = []

        if advantage:
            

            for TD_data in batch_data:
                # TD_size,{4}
                TD_reward = 0
                ob = TD_data[0]["ob"]
                act = TD_data[0]["action"]
                obp1 = TD_data[-1]["obp1"]

                for i in range(len(TD_data)):
                    TD_reward += TD_data[i]["reward"]

                done = TD_data[-1]["done"]
                discount = TD_data[-1]["discount"]

                # print("TD_reward:",TD_reward)

                obs.append(ob)
                acts.append(act)
                obp1s.append(obp1)
                rewards.append(TD_reward)
                dones.append(float(done))
                discounts.append(discount)

            return (
                np.array(
                    obs
                ),  # if self.backward_n_step > 1, [batch_size,backward_n_step+1,obaservation_size]
                np.array(acts),
                np.array(
                    obp1s
                ),  # if self.backward_n_step > 1, [batch_size,backward_n_step+1,obaservation_size]
                np.array(rewards),
                np.array(dones),
                np.array(discounts),
            )

        else:
            for TD_data in batch_data:
                # TD_size,{4}
                TD_reward = 0
                ob = TD_data[0]["ob"]

                for i in range(self.TD_size):
                    TD_reward += TD_data[i]["reward"]
                done = TD_data[-1]["done"]
                obs.append(ob)
                print("TD_reward:", TD_reward)
                rewards.append(TD_reward)
                dones.append(float(done))

            return np.array(obs), np.array(rewards), np.array(dones)

    def collect(self, must_be_whole_episode=True, prefill=False):
        """
        collect end when the play_records full or episode end
        """
        trainsaction = {
            "ob": copy.deepcopy(self.ob),
            "obp1": copy.deepcopy(self.ob),
            "reward": 0.0,
            "done": 0,
            "discount": np.array(1.0),
        }

        episode_record = [trainsaction]
        """
        I call this episode_record since when there's a done for playing, I must end collecting data for 
        capturing the end score of an episode, not to mix with next episode start when doing TD.
        It will start to train(in other words, break the loop) in two situation: 
        first, episode is done; second, the buffer is full(>self.batch_size * self.batch_length*self.TD_size).
        """
        # while len(episode_record) < self.batch_size * self.batch_length * self.TD_size:
        while True:  # stop only when episoe ends
            # episode = []
            # while True:

            if prefill:
                act = self.random_policy().numpy()  # to get batch dim
                # print("act:",act)

            else:

                obs_data = {
                    "obs": np.array([self.ob]),
                    "obp1s": np.array([self.ob]),
                    "rewards": 0.0,
                    "discounts": 1.0,
                }

                # print('tuple_of_episode_columns[2]:',np.amax(obs_data["obp1s"]))
                # print('tuple_of_episode_columns[2]:',np.amin(obs_data["obp1s"]))
                obs_data = {k: self._convert(v) for k, v in obs_data.items()}

                act, self.model.state = self.model.policy(obs_data, training=True)

                # print('tuple_of_episode_columns[2](after):',np.amax(obs_data["obp1s"]))
                # print('tuple_of_episode_columns[2](after):',np.amin(obs_data["obp1s"]))

                act = act.numpy()
                # print("act:",act)

            if self._c.is_discrete:
                argmax_act = np.argmax(act, axis=-1)

            # # save play img

            # if self.episode_step % 50000 == 0:
            #     self.save_play_img = True

            # if self.save_play_img == True:
            #     self.RGB_array_list.append(self.env.render())

            # if len(self.RGB_array_list) > 500:

            #     for i in range(len(self.RGB_array_list)):
            #         play_img = self.RGB_array_list[i]
            #         play_img = cv2.cvtColor(play_img, cv2.COLOR_RGB2BGR)

            #         cv2.imwrite("./train_gen_img/play_img_{}.jpg".format(i), play_img)
            #     self.RGB_array_list = []
            #     self.save_play_img = False

            ob, reward, done, info = self.act_repeat(
                self.env, argmax_act[0]
            )  # no repear any more

            if not prefill:
                self._step.assign_add(1)

            trainsaction = {
                "ob": copy.deepcopy(self.ob),
                "obp1": copy.deepcopy(ob),
                "action": copy.deepcopy(act[0]),
                "reward": copy.deepcopy(reward),
                "done": copy.deepcopy(done),
                "discount": np.array(
                    1 - float(done)
                ),  # it means when done, discount = 1, else 0.
            }  # ob+1(obp1) for advantage method
            # if self.episode_step < 50000:

            if self.episode_step >= self._c.time_limit:
                print("pre-set!")
                done = True
                trainsaction["done"] = done
                trainsaction["discount"] = np.array(1.0).astype(np.float32)

            episode_record.append(trainsaction)
            # print(
            #     "collecting_data!!:",
            #     self.episode_step,
            #     "len(episode_record):",
            #     len(episode_record),
            # )

            self.ob = ob

            # to coumpute average reward of each step
            self.episode_reward += reward
            self.episode_step += 1  # to avoid devide by zero
            self.total_step += 1
            # print("self.total_step:", self.total_step)

            if prefill == True:
                self.episode_step = 1
                # self.total_step = 1

            else:
                if self._step.numpy() % self._c.train_every == 0:
                    self.dreaming_update()
                    print("update complete!")
                else:
                    pass

            if done:
                print("game done!!")
                # self.env.reset()
                # self.ob, _, _, _ = self.env.step(
                #     self.env._env.action_space.sample()
                # )  # whether it is discrete or not, 0 is proper
                self.ob = self.env.reset()
                self.episode_step = 1

                average_reward = self.episode_reward / self.episode_step

                # to make first trasaction with zero action
                for key, value in episode_record[1].items():
                    if key not in episode_record[0]:
                        # print("episode_record[0] doesn't have key ", key)
                        episode_record[0][key] = 0 * value
                        # print("Now it is:",episode_record[0][key])

                if not prefill:
                    # for dreamer, it need to reset state at end of every episode
                    if self.model.state is not None and np.array([done]).any():
                        mask = tf.cast(1 - np.array([done]), self._float)[:, None]
                        self.model.state = tf.nest.map_structure(
                            lambda x: x * mask, self.model.state
                        )
                    else:
                        self.model.reset()

                break

        """
        move "TD_dict_to_TD_train_data" before update
        """
        if len(episode_record) > 1:
            # tranfer data to TD dict
            for i in range((len(episode_record) // self.TD_size)):

                TD_data = episode_record[i * self.TD_size : (i + 1) * self.TD_size]
                if (
                    len(TD_data) != self.TD_size
                ):  # to deal with +1 causing not enough data of a TD size
                    print("reverse take for taking just to end of episode")
                    TD_data = episode_record[-self.TD_size :]
                    import pdb
                    pdb.set_trace()
                self.play_records.append(TD_data)
                # so the structure of self.play_records is:
                # (batch_size* batch_length*TD_size or greater , TD_size)

            tuple_of_episode_columns = self.TD_dict_to_TD_train_data(
                self.play_records, True
            )  # for Dreamer, the TD = 1

            dict_of_episode_record = {
                "obs": tuple_of_episode_columns[0],
                "actions": tuple_of_episode_columns[1],
                "obp1s": tuple_of_episode_columns[2],
                "rewards": tuple_of_episode_columns[3],
                "dones": tuple_of_episode_columns[4],
                "discounts": tuple_of_episode_columns[5],
            }
            dict_of_episode_record = {
                k: self._convert(v) for k, v in dict_of_episode_record.items()
            }

            # reset the inner buffer
            filename = utils.save_episode(self.datadir, dict_of_episode_record)
            # if self.model.total_step % 100:

            self.post_process_episodes(self.episodes, filename, dict_of_episode_record)

            episode_record = []
            self.post_process_play_records()
        else:
            episode_record = []
            self.post_process_play_records()
            print("not enough data!!")

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)

    def post_process_play_records(self):
        self.play_records = []

    def post_process_episodes(self, cache, episode_name, episode):
        total = 0
        length = len(episode["rewards"]) - 1
        score = float(episode["rewards"].astype(np.float64).sum())
        video = episode["obp1s"]
        
        for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):

            if total <= self._c.max_dataset_steps - length:
                total += len(ep["rewards"]) - 1
            else:
                del cache[key]

        cache[str(episode_name)] = episode

        step = count_steps(self.datadir)

        with self._writer.as_default():
            tf.summary.scalar(
                "dataset_size",
                total + length,
                step=step * self._c.action_repeat,
            )  # control by model.total_step, record the env total step

            tf.summary.scalar(
                "train_episodes",
                len(cache),
                step=step * self._c.action_repeat,
            )  # control by model.total_step, record the env total step

            tf.summary.scalar(
                "train_return",
                score,
                step=step * self._c.action_repeat,
            )  # control by model.total_step, record the env total step

            tf.summary.scalar(
                "train_length",
                length,
                step=step * self._c.action_repeat,
            )  # control by model.total_step, record the env total step

            print("save train_policy!!!!")
            tools.video_summary(
                "train_policy", np.array(video[None]), step * self._c.action_repeat
            )

        print("the episodes size now is:", total + length, "steps")

    # @tf.function
    def dreaming_update(self):
        """
        using collect function
        1. collect real play record(this should from collect function)
           get (batch,batch_length,feature_size) data

        when geting enough data
        2. using the real play record, batch by batch process
        3. in each batch process:
            world model stage
            a. do extract embed from data
            b. from embed to feat
            c. using feat to reconstruct observation
            d. get world model gradient

            actor stage
            e. do imagine step to horizon, for actor and critic:
            for example:
                from (batch,batch_length,feature_size)  to (batch, horizon,batch_length,feature_size)
            f. get value
            g. do dreamer lambda return
            h. maximize lambda return

            critic stage
            i. make value and lambda return closer as much as possible


        4. in each batch process, after imagine step, do update actor and critic
        """
        data = next(self._dataset)

        data = utils.preprocess(data, self._c)

        obs, actions, obp1s, rewards, dones, discounts = (
            data["obs"],
            data["actions"],
            data["obp1s"],
            data["rewards"],
            data["dones"],
            data["discounts"],
        )

        # print("obs:", obs.shape)  # (50, 10, 160, 160, 1)
        # print("actinons", actions.shape)
        # print("obp1s:", obs.shape)  # (50, 10, 160, 160, 1)
        # print("rewards:", rewards.shape)  # (50, 10)
        # print("dones:", rewards.shape)  # (50, 10)
        # print("discounts:", discounts.shape)  # (50, 10)

        start_time = time.time()
        rewards_mean = self.model.update_dreaming(
            obs, actions, obp1s, rewards, dones, discounts
        )

        # self._step.assign_add(len(data["dones"]))

        end_time = time.time()
        # print("update time = ", end_time - start_time)
        return rewards_mean

    def reset_total_step(self):
        """
        When prefill end, call this function
        """
        self.total_step = 1