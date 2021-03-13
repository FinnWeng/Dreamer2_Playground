import tensorflow as tf
import tools
import numpy as np
import cv2
import net.layers as layers


# class ActionDecoder(tf.Module):
#     def __init__(self, action_dim, layers, filters):
#         self.action_dim = action_dim
#         self._layers = layers
#         self.filters = filters

#     def get(self, name, ctor, *args, **kwargs):
#         # Create or get layer by name to avoid mentioning it in the constructor.
#         if not hasattr(self, "_modules"):
#             self._modules = {}
#         if name not in self._modules:
#             self._modules[name] = ctor(*args, **kwargs)
#         return self._modules[name]

#     def __call__(self, features):
#         x = features
#         for index in range(self._layers):

#             if index / 2 == 0:
#                 x = self.get(f"h{index}", tf.keras.layers.Dense, self.filters)(x)
#             else:
#                 x = self.get(
#                     f"h{index}",
#                     tf.keras.layers.Dense,
#                     self.filters,
#                     tf.keras.layers.PReLU(),
#                 )(x)

#         # links = tf.keras.layers.Dense(self.action_dim - 1,activation="tanh")(x) # act = 1~-1
#         links = self.get("links", tf.keras.layers.Dense, self.action_dim - 1, "tanh")(x)
#         # gripper = tf.keras.layers.Dense(1,activation="sigmoid")(x) # act = 0~1
#         gripper = self.get("gripper", tf.keras.layers.Dense, 1, "sigmoid")(x)
#         # print("links:",links)
#         # print("gripper:",gripper)
#         output = tf.concat([links, gripper], axis=-1)
#         return output


class ActionDecoder(tf.keras.Model):
    def __init__(self, action_dim, layers_num, filters):
        super(ActionDecoder, self).__init__()
        # self.hidden_size = hidden_size
        self.action_dim = action_dim
        self._layers_num = layers_num
        self.filters = filters
        self.layers_list = []
        for index in range(self._layers_num):
            if index / 2 == 0:
                self.layers_list.append(tf.keras.layers.Dense(self.filters))
            else:
                self.layers_list.append(
                    tf.keras.layers.Dense(self.filters, tf.keras.layers.PReLU())
                )

        self.links_layers = tf.keras.layers.Dense(self.action_dim - 1, "tanh")
        self.gripper_layers = tf.keras.layers.Dense(1, "sigmoid")

    def __call__(self, features):
        x = features

        for layers_fn in self.layers_list:
            x = layers_fn(x)

        links = self.links_layers(x)
        gripper = self.gripper_layers(x)
        output = tf.concat([links, gripper], axis=-1)

        return output


# class QDecoder(tf.Module):
#     def __init__(self, layers, filters):

#         self._layers = layers
#         self.filters = filters

#     def get(self, name, ctor, *args, **kwargs):
#         # Create or get layer by name to avoid mentioning it in the constructor.
#         if not hasattr(self, "_modules"):
#             self._modules = {}
#         if name not in self._modules:
#             self._modules[name] = ctor(*args, **kwargs)
#         return self._modules[name]

#     def __call__(self, features, action):
#         x = features
#         x = tf.concat([x, action], -1)
#         for index in range(self._layers):

#             if index / 2 == 0:
#                 x = self.get(f"h{index}", tf.keras.layers.Dense, self.filters)(x)
#             else:
#                 x = self.get([[[]]]
#                     f"h{index}",
#                     tf.keras.layers.Dense,
#                     self.filters,
#                     tf.keras.layers.PReLU(),
#                 )(x)

#         x = self.get("out1", tf.keras.layers.Dense, self.filters)(x)
#         x = self.get("out2", tf.keras.layers.Dense, 1)(x)

#         return x


class QDecoder(tf.keras.Model):
    def __init__(self, layers_num, filters):
        super(QDecoder, self).__init__()
        # self.hidden_size = hidden_size
        self._layers_num = layers_num
        self.filters = filters

        self.layers_list = []
        for index in range(self._layers_num):
            if index / 2 == 0:
                self.layers_list.append(tf.keras.layers.Dense(self.filters))
            else:
                self.layers_list.append(
                    tf.keras.layers.Dense(self.filters, tf.keras.layers.PReLU())
                )
        self.out1 = tf.keras.layers.Dense(self.filters)
        self.out2 = tf.keras.layers.Dense(1)

    def __call__(self, features, action):
        x = features
        x = tf.concat([x, action], -1)
        for layers_fn in self.layers_list:
            x = layers_fn(x)

        x = self.out1(x)
        x = self.out2(x)

        return x


class DDPG:
    def __init__(self, env):
        # define callback and set networ parameters
        gpus = tf.config.experimental.list_physical_devices("GPU")

        tf.config.experimental.set_visible_devices(gpus[1], "GPU")
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        # for discrete space, the space means kinds of choice of action. for continious, it means upper and lower bound of value.
        # so the action_dim means number of action. it is different from for discrete space, kinds of choice.
        self.env = env
        self.action_dim = env.action_dim
        # self.observation_dim = env.observation_dim # 57
        self.observation_dim = len(env._observation)
        self.filters = 4
        self.kernel_size = 3
        self.activation_fn = tf.keras.layers.PReLU
        self.hidden_size = 20

        self.lr = 1e-5
        self.gamma = 0.99
        self.tau = 0.001

        self.adj = [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
        ]

        self.DDPG_enc_path = "./model/DDPG_enc.ckpt"
        self.DDPG_actor_path = "./model/DDPG_actor.ckpt"
        self.DDPG_critic_path = "./model/DDPG_critic.ckpt"

        self.GAT = layers.GAT(
            nb_classes=3,
            nb_nodes=8,
            training=True,
            attn_drop=0.6,
            ffd_drop=0.6,
            bias_mat=self.adj,
            hid_units=[8],
            n_heads=[8, 1],
            activation=tf.nn.elu,
            residual=False,
        )
        self.encoder = self.make_G_encoder()
        self.target_encoder = self.make_G_encoder()

        # self.encoder = self.make_encoder()
        # self.target_encoder = self.make_encoder()
        self.actor = ActionDecoder(self.action_dim, 4, self.filters * 4)
        self.target_actor = ActionDecoder(self.action_dim, 4, self.filters * 4)
        self.critic = QDecoder(4, self.filters * 4)
        self.target_critic = QDecoder(4, self.filters * 4)

        self.encoder_tffn = tf.function(self.encoder)
        self.actor_tffn = tf.function(self.actor)
        self.critic_tffn = tf.function(self.critic)

        self.actor_optimizer = tf.optimizers.Adam(self.lr)
        self.critic_optimizer = tf.optimizers.Adam(self.lr)

        self.random_policy_playing = self.random_policy

        self._writer = tf.summary.create_file_writer(
            "./tf_log", max_queue=1000, flush_millis=20000
        )
        self.total_step = 1
        self.save_play_img = False

        self.RGB_array_list = []

    def make_encoder(self):
        """
        since the observation 3(pos)+3(euler orn) for one link, the input channels should be 6 or multiple of 6.
        the last 3 columns is relative position of block, so it is been spilit out and concat to latent later
        """
        links_input = tf.keras.Input(
            shape=[self.observation_dim], name="image_input", dtype=tf.float32
        )
        base_orn = links_input[:, -4:]
        multi_channel_input = tf.reshape(
            links_input[:, :-4], [-1, 8, (self.observation_dim - 4) // 8]
        )  # -1, 8,3
        multi_channel_input = tf.transpose(multi_channel_input, [0, 2, 1])  # -1, 8, 3
        x = tf.keras.layers.Conv1D(
            self.filters, kernel_size=2, strides=1, padding="valid", use_bias=False,
        )(
            multi_channel_input
        )  # -1,7,4
        x = tf.keras.layers.Conv1D(
            self.filters,
            kernel_size=1,
            strides=1,
            padding="valid",
            activation=self.activation_fn(),
            use_bias=False,
        )(
            x
        )  # -1,7 ,4
        x = tf.keras.layers.Conv1D(
            self.filters * 2,
            kernel_size=self.kernel_size,
            strides=2,
            padding="valid",
            use_bias=False,
        )(
            x
        )  # -1,3 ,8
        x = tf.keras.layers.Conv1D(
            self.filters * 4,
            kernel_size=1,
            strides=1,
            padding="valid",
            activation=self.activation_fn(),
            use_bias=False,
        )(
            x
        )  # -1,3 ,16

        # flatten
        x = tf.keras.layers.Conv1D(
            self.filters * 4,
            kernel_size=x.shape[-2],
            strides=1,
            padding="valid",
            use_bias=False,
        )(
            x
        )  # -1,1 ,16
        x = tf.reshape(x, [-1, self.filters * 4])  # [-1,16]
        x = tf.concat([x, base_orn], -1)  # [-1, 19]
        # print("encoder x:",x)

        x = tf.keras.layers.Dense(self.hidden_size)(x)  # [-1,20]

        encoder_model = tf.keras.Model(inputs=[links_input], outputs=[x])

        return encoder_model

    def make_G_encoder(self):
        links_input = tf.keras.Input(
            shape=[self.observation_dim], name="image_input", dtype=tf.float32
        )
        blockPosInGripper = links_input[:, -4:]
        multi_channel_input = tf.reshape(
            links_input[:, :-4], [-1, (self.observation_dim - 4) // 3, 3]
        )  # -1, 9,6
        # x = tf.keras.layers.Conv1D(self.filters,kernel_size=self.kernel_size,strides=1,padding="valid",use_bias=False)(multi_channel_input) # -1,7,4
        # x = tf.keras.layers.Conv1D(self.filters,kernel_size=1,strides=1,padding="valid",activation=self.activation_fn(),use_bias=False)(x) # -1,7 ,4
        # x = tf.keras.layers.Conv1D(self.filters*2,kernel_size=self.kernel_size,strides=2,padding="valid",use_bias=False)(x) # -1,3 ,8
        # x = tf.keras.layers.Conv1D(self.filters*4,kernel_size=1,strides=1,padding="valid",activation=self.activation_fn(),use_bias=False)(x) # -1,3 ,16

        print("multi_channel_input:", multi_channel_input.shape)
        x = self.GAT(multi_channel_input)

        # flatten
        x = tf.keras.layers.Conv1D(
            self.filters * 4,
            kernel_size=x.shape[-2],
            strides=1,
            padding="valid",
            use_bias=False,
        )(
            x
        )  # -1,1 ,16
        x = tf.reshape(x, [-1, self.filters * 4])  # [-1,16]
        x = tf.concat([x, blockPosInGripper], -1)  # [-1, 19]
        # print("encoder x:",x)

        encoder_model = tf.keras.Model(inputs=[links_input], outputs=[x])

        return encoder_model

    def random_policy(self):
        return tf.random.uniform(
            [1, self.action_dim], minval=-1, maxval=1, dtype=tf.dtypes.float32
        )

    def doubl_trick_update(self, target, origin):
        for (layer_target, layer_origin) in zip(target.layers, origin.layers):
            origin_weights = layer_origin.get_weights()
            target_weights = layer_target.get_weights()
            setting_weights = [
                (1 - self.tau) * target_weight + self.tau * origin_weight
                for target_weight, origin_weight in zip(target_weights, origin_weights)
            ]
            layer_target.set_weights(setting_weights)

    def update(self, obs, rewards):

        with tf.GradientTape() as actor_tape:

            embed = self.encoder(obs)
            # print(embed)
            action = self.actor(embed)
            print("action.shape:", action.shape)
            predict_Q = tf.stop_gradient(self.critic(embed, action))
            print("predict_Q:", predict_Q)
            # maximize this Q

        with tf.GradientTape() as critic_tape:
            embed = self.encoder(obs)
            # print(embed)
            action = self.actor(embed)
            print("action.shape:", action.shape)
            predict_Q = self.critic(embed, tf.stop_gradient(action))
            print("predict_Q:", predict_Q)
            # minimize Q - Qtrue

        return 0

    def update_advantage(self, obs, obp1s, rewards, dones):

        with tf.GradientTape() as actor_tape:

            embed = self.encoder_tffn(obs)
            # print(embed)
            action = self.actor_tffn(embed)
            # print("action.shape:", action.shape) # (4, 8)
            predict_Q = self.critic_tffn(embed, action)
            actor_loss = -1 * tf.reduce_mean(predict_Q)  # minus for maximize
            # print("predict_Q:", predict_Q) # (4, 1)
            # maximize this Q

        actor_var = []
        actor_var.extend(self.encoder.trainable_variables)
        actor_var.extend(self.actor.trainable_variables)
        actor_grads = actor_tape.gradient(actor_loss, actor_var)
        # print("actor grad:")
        # for g in actor_grads:
        #     print(g.shape)

        with tf.GradientTape() as critic_tape:
            embed = self.encoder_tffn(obs)
            # print(embed)
            action = self.actor_tffn(embed)
            predict_Q = self.critic_tffn(embed, tf.stop_gradient(action))

            embed_p1 = self.target_encoder(obp1s)
            action_p1 = self.target_actor(embed_p1)
            predict_Q_p1 = tf.stop_gradient(self.target_critic(embed_p1, action_p1))

            # minimize Q - Qtrue
            target_Q = rewards + (1 - dones) * self.gamma * predict_Q_p1
            critic_loss = tf.reduce_mean(tf.square(target_Q - predict_Q))

        critic_var = []
        critic_var.extend(self.encoder.trainable_variables)
        critic_var.extend(self.critic.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, critic_var)
        # print("critic grad:")
        # for g in critic_grads:
        #     print(g.shape)

        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_var))
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_var))

        print("predict_Q.numpy().mean():", predict_Q.numpy().mean())
        # print("rewards:",rewards.mean())

        print("save_play_img:", self.save_play_img)
        print("self.RGB_array_list:", len(self.RGB_array_list))
        print("self.total_step", self.total_step)
        if self.total_step % 10000 == 0:
            self.encoder.save_weights(self.DDPG_enc_path)
            self.actor.save_weights(self.DDPG_actor_path)
            self.critic.save_weights(self.DDPG_critic_path)

            # self.save_play_img = True
        # if self.save_play_img == True:
        #     self.RGB_array_list.append(self.env.render())
        # if len(self.RGB_array_list) > 500:
        #     tools.graph_summary(
        #         self._writer,
        #         tools.video_summary,
        #         "pybullet_play/pybullet_play",
        #         np.array(self.RGB_array_list),
        #     )
        #     for i in range(len(self.RGB_array_list)):
        #         play_img = self.RGB_array_list[i]
        #         play_img = cv2.cvtColor(play_img, cv2.COLOR_RGB2BGR)

        #         cv2.imwrite("./train_gen_img/play_img_{}.jpg".format(i), play_img)
        #     self.RGB_array_list = []
        #     self.save_play_img = False

        self.total_step += 1
        tf.summary.experimental.set_step(self.total_step)

        # include double trick
        self.doubl_trick_update(self.target_encoder, self.encoder)
        self.doubl_trick_update(self.target_actor, self.actor)
        self.doubl_trick_update(self.target_critic, self.critic)

        return rewards.mean()

    def save_summary(self, average_reward):
        tf.summary.scalar("average_reward", average_reward, step=self.total_step)

