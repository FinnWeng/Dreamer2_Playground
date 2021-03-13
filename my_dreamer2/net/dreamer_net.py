import tensorflow as tf
import tools
import numpy as np
import cv2
import net.layers as layers
from tensorflow_probability import distributions as tfd
import collections
import utils


def world_model_imagine(fn, state_dict, action_array):
    """
    from state, input (action)
    state(stoch): (25*batch_length, 30)
    action_array: (batch_size, horizon,act_dim)
    """
    state_dict_list = []
    current_state = state_dict
    output_dict = {}
    for i in range(action_array.shape[1]):  # to horizon
        current_state = fn(current_state, action_array[:, i, :])
        state_dict_list.append(current_state)
    for k, v in state_dict.items():
        list_of_v = [
            x[k] for x in state_dict_list
        ]  # each is (batch_size*batch_length, 30)
        v_tensor = tf.stack(
            list_of_v, 1
        )  # each is (batch_size*batch_length, horizon, 30)
        output_dict[k] = v_tensor

    return output_dict


def world_model_observing(fn, state_dict, action_array, embed_array):
    """
    from state, input (embed and action)
    state(stoch): (25, 30)
    embed: (25, 50, 1024)
    action: (25, 50, 4)
    action_array: (batch_size, horizon,act_dim)
    """
    current_state = state_dict
    post_output_dict = {}
    prior_output_dict = {}
    post_dict_list = []
    prior_dict_list = []
    current_state = state_dict
    output_dict = {}

    for i in range(action_array.shape[1]):
        # print("action_array[:, i, :]:", action_array[:, i, :].shape) # (batch_size,4)
        # print("embed_array[:, i, :]", embed_array[:, i, :].shape) # (batch_size,1024)
        post_state, prior_state = fn(
            current_state, action_array[:, i, :], embed_array[:, i, :]
        )
        post_dict_list.append(post_state)
        prior_dict_list.append(prior_state)
    for k, v in state_dict.items():
        post_v_list = [
            x[k] for x in post_dict_list
        ]  # each is (25, 30), have batch_length entries
        prior_v_list = [
            x[k] for x in prior_dict_list
        ]  # each is (25, 30) , have batch_length entries
        post_v_tensor = tf.stack(post_v_list, 1)  # (25, batch_length,30)
        prior_v_tensor = tf.stack(prior_v_list, 1)  # (25, batch_length,30)
        post_output_dict[k] = post_v_tensor
        prior_output_dict[k] = prior_v_tensor

    return post_output_dict, prior_output_dict


class ConvEncoder(tf.keras.Model):
    def __init__(self, filters, activation=tf.nn.relu):
        super(ConvEncoder, self).__init__()

        self.activation = activation
        self.filters = filters
        kwargs = dict(strides=2, activation=self.activation)
        self.l1 = tf.keras.layers.Conv2D(1 * self.filters, kernel_size=4, **kwargs)
        self.l2 = tf.keras.layers.Conv2D(2 * self.filters, kernel_size=4, **kwargs)
        self.l3 = tf.keras.layers.Conv2D(4 * self.filters, kernel_size=4, **kwargs)
        self.l4 = tf.keras.layers.Conv2D(8 * self.filters, kernel_size=4, **kwargs)
        # self.l5 = tf.keras.layers.Conv2D(8 * self.filters, kernel_size=4, **kwargs)

    def __call__(self, obs):
        # obs: (batch_size, batch_length, h,w,c)

        x = tf.reshape(
            obs, (-1,) + tuple(obs.shape[-3:])
        )  # (1, 64, 64, 1) or ( 1 ,1, 64, 64, 1)

        x = self.l1(x)  # (1, 31, 31, 32)

        x = self.l2(x)  # (1, 14, 14, 64)

        x = self.l3(x)  # (1, 6, 6, 128)
        x = self.l4(x)  # (1, 2, 2, 256)
        # x = self.l5(x)

        hw_shape = tf.shape(x)[1:3]

        out_shape = tf.concat(
            [tf.shape(obs)[:-3], [hw_shape[0] * hw_shape[1] * 8 * self.filters]], 0
        )  # 8*2*2 = 32

        return tf.reshape(x, out_shape)  # tf.Tensor([   1    1 1024]


class ConvDecoder(tf.keras.Model):
    def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
        super(ConvDecoder, self).__init__()
        self._act = act
        self._depth = depth
        self._shape = shape

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]

    def __call__(self, features):
        kwargs = dict(strides=2, activation=self._act)
        x = self.get("h1", tf.keras.layers.Dense, 32 * self._depth, None)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        x = self.get(
            "h2", tf.keras.layers.Conv2DTranspose, 4 * self._depth, 5, **kwargs
        )(x)
        x = self.get(
            "h3", tf.keras.layers.Conv2DTranspose, 2 * self._depth, 5, **kwargs
        )(x)
        x = self.get(
            "h4", tf.keras.layers.Conv2DTranspose, 1 * self._depth, 6, **kwargs
        )(x)
        x = self.get(
            "h5", tf.keras.layers.Conv2DTranspose, self._shape[-1], 6, strides=2
        )(x)
        mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class ActionDecoder(tf.keras.Model):
    def __init__(
        self,
        action_dim,
        layers_num,
        filters,
        dist="tanh_normal",
        act=tf.nn.elu,
        min_std=1e-4,
        init_std=5,
        mean_scale=5,
    ):
        super(ActionDecoder, self).__init__()
        # self.hidden_size = hidden_size
        self.action_dim = action_dim
        self._layers_num = layers_num
        self.filters = filters
        self._size = action_dim
        self.layers_num = layers_num
        self._units = filters
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]

    def __call__(self, features):
        raw_init_std = np.log(np.exp(self._init_std) - 1)
        x = features
        for index in range(self._layers_num):
            x = self.get(f"h{index}", tf.keras.layers.Dense, self._units, self._act)(x)
        if self._dist == "tanh_normal":
            # https://www.desmos.com/calculator/rcmcf5jwe7
            x = self.get(f"hout", tf.keras.layers.Dense, 2 * self._size)(x)
            mean, std = tf.split(x, 2, -1)
            mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
            std = tf.nn.softplus(std + raw_init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "onehot":
            x = self.get(f"hout", tf.keras.layers.Dense, self._size)(x)
            dist = tools.OneHotDist(x)
        else:
            raise NotImplementedError(dist)
        return dist


class ValueDecoder(tf.keras.Model):
    def __init__(self, shape, layers_num, filters, dist="normal", act=tf.nn.elu):
        super(ValueDecoder, self).__init__()
        # (), 3, self._c.num_units = 400, act=act
        self._shape = shape
        self.layers_num = layers_num
        self._units = filters
        self._dist = dist
        self._act = act

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]

    def __call__(self, features):
        x = features
        for index in range(self.layers_num):  # 3
            x = self.get(f"h{index}", tf.keras.layers.Dense, self._units, self._act)(x)
            # print("x:",x.shape) #  (15, 1250, 400)
        x = self.get(f"hout", tf.keras.layers.Dense, np.prod(self._shape))(x)
        # print("x:",x.shape)
        x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        # print("x:",x.shape)
        if self._dist == "normal":
            return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
        if self._dist == "binary":
            return tfd.Independent(tfd.Bernoulli(x), len(self._shape))
        raise NotImplementedError(self._dist)


class RSSM(tf.keras.Model):
    def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
        super(RSSM, self).__init__()
        self._activation = act
        self._stoch_size = stoch
        self._deter_size = deter
        self._hidden_size = hidden
        self._cell = tf.keras.layers.GRUCell(
            self._deter_size
        )  # # (output, next_state) = call(input, state)

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]

    def initial(self, batch_size):
        dtype = tf.float32
        return dict(
            mean=tf.zeros([batch_size, self._stoch_size], dtype),
            std=tf.zeros([batch_size, self._stoch_size], dtype),
            stoch=tf.zeros([batch_size, self._stoch_size], dtype),
            deter=self._cell.get_initial_state(None, batch_size, dtype),
        )  # zero initialization, float32

    def get_feat(self, state):
        """
        which means:
        some of features are stochastic, some are deterministic
        stoch is sampled from post of observation step of imagination
        deterministic is from RSSM's GRUcell
        """
        return tf.concat([state["stoch"], state["deter"]], -1)

    def get_dist(self, state):
        return tfd.MultivariateNormalDiag(state["mean"], state["std"])

    @tf.function
    def img_step(self, prev_state, prev_action):
        """
        the basic step, run without obs. the one with obs bases on this function
        p(st|st-1,at-1)
        """
        # the image step from St-1 to St (be aware that this is not OBSERVATION)
        print()
        x = tf.concat([prev_state["stoch"], prev_action], -1)  # => (25, 34)
        x = self.get(
            "img1", tf.keras.layers.Dense, self._hidden_size, self._activation
        )(x)
        x, deter = self._cell(
            x, [prev_state["deter"]]
        )  # (output, next_state) = call(input, state)
        # x: (25, 200)

        deter = deter[0]  # Keras wraps the state in a list.

        # below is VAE prior, which means P(z|x)
        x = self.get(
            "img2", tf.keras.layers.Dense, self._hidden_size, self._activation
        )(x)
        x = self.get("img3", tf.keras.layers.Dense, 2 * self._stoch_size, None)(
            x
        )  # =>(25, 60)

        mean, std = tf.split(
            x, 2, -1
        )  # =>(25, 30), =>(25, 30) or =>(1250, 30), =>(1250, 30), the 1250 is 25*50 since the flatten function of imagine head
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_dist({"mean": mean, "std": std}).sample()
        print("stoch:", stoch.shape)  # (25, 30)

        """
        deter is purely from neural network, which is deterministic.
        stoch is sample from distribution decided by deter, which is stochastic.
        """

        prior = {"mean": mean, "std": std, "stoch": stoch, "deter": deter}
        return prior

    @tf.function
    def obs_step(self, prev_state, prev_action, embed):
        """
        p(st|st-1,at-1,ot)
        """
        prior = self.img_step(prev_state, prev_action)  # get the prior of VAE
        # print('prior["deter"]:', prior["deter"].shape)  # (batch_size, 200)

        # below gets posterior of VAE
        x = tf.concat([prior["deter"], embed], -1)  # concat state and observation
        x = self.get(
            "obs1", tf.keras.layers.Dense, self._hidden_size, self._activation
        )(x)
        x = self.get("obs2", tf.keras.layers.Dense, 2 * self._stoch_size, None)(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_dist({"mean": mean, "std": std}).sample()
        post = {"mean": mean, "std": std, "stoch": stoch, "deter": prior["deter"]}
        return post, prior

    @tf.function
    def imagine(self, action, state=None):  # been used only for summary
        if state is None:
            # stoch=tf.zeros([batch_size, self._stoch_size], dtype)
            state = self.initial(tf.shape(action)[0])

        prior = world_model_imagine(
            self.img_step, state, action
        )  # each is (batch_length*batch_length, horizon, 30)

        return prior

    @tf.function
    def observe(self, embed, action, state=None):
        """
        using obs_step to acturely do observe since it need to do transpose
        """
        # embed  (25, 50, 1024)
        # action (25, 50, 4)
        if state is None:
            state = self.initial(tf.shape(action)[0])

        post, prior = world_model_observing(
            self.obs_step, state, action, embed
        )  # each value of dict is (25, batch_length,30)
        return post, prior


class Dreamer:
    def __init__(self, env, is_training, config):

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
        self._c = config
        actspace = self.env.action_space
        self._actspace = actspace
        self._actdim = actspace.n if hasattr(actspace, "n") else actspace.shape[0]

        actvs = dict(
            elu=tf.nn.elu,
            relu=tf.nn.relu,
            swish=tf.nn.swish,
            leaky_relu=tf.nn.leaky_relu,
        )
        cnn_actv = actvs[self._c.cnn_act]
        actv = actvs[self._c.dense_act]
        self._metrics = collections.defaultdict(tf.metrics.Mean)
        self._metrics["expl_amount"]  # Create variable for checkpoint.

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

        self.Dreamer_dynamics_path = "./model/Dreamer_dynamics.ckpt"
        self.Dreamer_encoder_path = "./model/Dreamer_encoder.ckpt"
        self.Dreamer_reward_decoder_path = "./model/Dreamer_reward_decoder.ckpt"
        self.Dreamer_critic_path = "./model/Dreamer_critic.ckpt"
        self.Dreamer_actor_path = "./model/Dreamer_actor.ckpt"

        self.dynamics = RSSM(self._c.stoch_size, self._c.deter_size, self._c.deter_size)

        self.encoder = ConvEncoder(self._c.cnn_depth, cnn_actv)
        if self._c.grayscale:
            output_shape = (64, 64, 1)
        else:
            output_shape = (64, 64, 3)

        self.decoder = ConvDecoder(self._c.cnn_depth, cnn_actv, shape=output_shape)

        self.reward_decoder = ValueDecoder((), 2, self._c.num_units, act=actv)
        if self._c.pcont:
            self._pcont = ValueDecoder((), 3, self._c.num_units, "binary", act=actv)
            print("self._pcont:", self._pcont.variables)
        self.critic = ValueDecoder(
            (), 3, self._c.num_units, act=actv
        )  # critic, no advantage

        self.actor = ActionDecoder(
            self._actdim,
            4,
            self._c.num_units,
            self._c.action_dist,
            init_std=self._c.action_init_std,
            act=actv,
        )

        model_modules = [self.encoder, self.dynamics, self.decoder, self.reward_decoder]
        self.state = None

        # self.encoder_tffn = tf.function(self.encoder)
        # self.actor_tffn = tf.function(self.actor)
        # self.critic_tffn = tf.function(self.critic)

        self.world_optimizer = tf.optimizers.Adam(self._c.model_lr)
        self.actor_optimizer = tf.optimizers.Adam(self._c.actor_lr)
        self.critic_optimizer = tf.optimizers.Adam(self._c.value_lr)

        self._writer = tf.summary.create_file_writer(
            "./tf_log", max_queue=1000, flush_millis=20000
        )
        self.total_step = 1
        # self.total_step = int(15.66*1000)
        self.save_play_img = False

        self.RGB_array_list = []

        model_list = [self.dynamics,self.encoder,self.reward_decoder, self.critic, self.actor]
        model_ckpt_path_list = [self.Dreamer_dynamics_path,self.Dreamer_encoder_path,self.Dreamer_reward_decoder_path,self.Dreamer_critic_path, self.Dreamer_actor_path]

        if not is_training:
            self.dynamics.load_weights(self.Dreamer_dynamics_path)
            self.encoder.load_weights(self.Dreamer_encoder_path)
            # self.decoder
            # self.reward_decoder.load_weights(self.Dreamer_reward_decoder_path)
            # self.critic.load_weights(self.Dreamer_critic_path)
            self.actor.load_weights(self.Dreamer_actor_path)

            # self.Dreamer_dynamics_path = "./model/Dreamer_dynamics.ckpt"
            # self.Dreamer_encoder_path = "./model/Dreamer_encoder.ckpt"
            # self.Dreamer_reward_decoder_path = "./model/Dreamer_reward_decoder.ckpt"
            # self.Dreamer_critic_path = "./model/Dreamer_critic.ckpt"
            # self.Dreamer_actor_path = "./model/Dreamer_actor.ckpt"
        if not is_training:


    def reset(self):
        # this reset the state saved for RSSM forwarding
        self.state = None

    def _exploration(self, action, training):
        if training:
            amount = self._c.expl_amount
            if self._c.expl_decay:  # 0.3 is True
                amount *= 0.5 ** (tf.cast(self._step, tf.float32) / self._c.expl_decay)
            if self._c.expl_min:  # 0.0 is False
                amount = tf.maximum(self._c.expl_min, amount)
            self._metrics["expl_amount"].update_state(amount)
        elif self._c.eval_noise:
            amount = self._c.eval_noise
        else:
            return action
        if self._c.expl == "additive_gaussian":
            return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
        if self._c.expl == "completely_random":
            return tf.random.uniform(action.shape, -1, 1)
        if self._c.expl == "epsilon_greedy":
            # print(
            #     "0 * action:", 0 * action
            # )  # 0 * action: Tensor("mul:0", shape=(1, 4), dtype=float16)
            indices = tfd.Categorical(0 * action).sample()
            # print("epsilon_greedy indices:", indices)  # shape=(1,)
            return tf.where(
                tf.random.uniform(action.shape[:1], 0, 1)
                < amount,  # randomly decide do epsilon greedy or not
                tf.one_hot(indices, action.shape[-1], dtype=tf.float32),
                action,
            )

        raise NotImplementedError(self._c.expl)

    def policy(self, obs, training=False):
        # this combine the encoder and actor and get feat to get a totoal agent policy
        if self.state is None:
            latent = self.dynamics.initial(len(obs))
            action = tf.zeros((len(obs), self._actdim), tf.float32)
        else:
            latent, action = self.state
        obs = tf.cast(obs, tf.float32)
        embed = self.encoder(obs)
        latent, _ = self.dynamics.obs_step(
            latent, action, embed
        )  # using post, which using future state to get action # need to to slicing since here should using scan
        feat = self.dynamics.get_feat(latent)
        if training:
            action = self.actor(feat).sample()
        else:
            action = self.actor(feat).mode()

        action = self._exploration(action, training)

        if self._c.is_discrete:
            action = tf.nn.softmax(action, axis=-1)

        state = (latent, action)
        return action, state

    def imagine_ahead(self, start_state):
        # each value of start_state(post) dict is (25, batch_length,30)
        state = start_state
        # this is different from function imagine. this use fresh action from policy to get next state, and next state.
        flatten = lambda x: tf.reshape(
            x, [-1] + list(x.shape[2:])
        )  # to (batch_size*batch_length,230)
        start = {k: flatten(v) for k, v in state.items()}  # flatten evey entry of dict
        policy = lambda state: self.actor(
            tf.stop_gradient(self.dynamics.get_feat(state))
        ).sample()  # local policy since we don't want to update world model here.

        img_step_fresh_action = lambda prev, _: self.dynamics.img_step(
            prev, policy(prev)
        )  # the _ is where the storage action is been put
        control_array = tf.range(
            self._c.horizon
        )  # now ths action is only for control iteration times. So I called it control array
        control_array = tf.reshape(control_array, [1, -1, 1])
        states = world_model_imagine(
            img_step_fresh_action, start, control_array
        )  # each is (batch_length*batch_length, horizon, 30)

        imag_feat = self.dynamics.get_feat(
            states
        )  # concate state and obs # (1225,15,  230)

        return imag_feat

    def lambda_returns(self, img_feat, pcont, _lambda):
        """
        it is extend from imagine_ahead
        there're three type of value estimation for imagine steps:
        type 1. prurely use imagine reward from world model.
        type 2. decide a k for imagine steps. befor k, using world model, after k, use critic
        type 3. consider all k and take sum of all type 2 till H - 1. use critic for H(orizon) step
        """
        # img_feat = # (1225,15, 230)

        reward_pred = self.reward_decoder(
            img_feat
        ).mode()  # reward_pred.sample(): (25*batch_length,horizon-1)

        value = self.critic(img_feat).mode()  # (25*batch_length,horizon)

        value = tf.transpose(value, [1, 0])
        reward_pred = tf.transpose(reward_pred, [1, 0])
        pcont = tf.transpose(pcont, [1, 0])
        # print("value:",value.shape) # (horizon, 25*batch_length)
        # print("reward_pred:",reward_pred.shape)# (horizon, 25*batch_length)
        # print("pcont:",pcont.shape)

        # # now try to assemble above to become lambda reward
        # # reward_pcont = tf.concat([tf.ones_like(pcont[0]), pcont[:-1]], 0)
        # # discount_reward_pred = [
        # #     reward_pred[i, :] * reward_pcont for i in range(self._c.horizon)]
        # # discount_value = [value[i, :] * pcont for i in range(self._c.horizon)]

        # discount_reward_pred = [
        #     reward_pred[i, :] * pcont[i,:] for i in range(self._c.horizon)]
        # discount_value = [value[i, :] * pcont[i,:] for i in range(self._c.horizon)]

        # # accu_discount_reward_pred = []
        # # for i in range(self._c.horizon):
        # #     accu_discount_reward_pred.append(discount_reward_pred[:i])
        # # accu_discount_reward_pred = [
        # #     tf.reduce_sum(tf.stack(accu_discount_reward_pred[j], -1), 0)
        # #     * _lambda ** (i)
        # #     for j in range(self._c.horizon)
        # # ]  # sum for type 2, which to k step
        # # accu_discount_reward_pred = tf.stack(
        # #     accu_discount_reward_pred, axis=0
        # # )  # (15,1225) , for each reward accumulate to n step.

        # accu_discount_reward_pred = tf.stack(
        #     accu_discount_reward_pred, axis=0
        # )  # (15,1225) , for each reward accumulate to n step.

        # type2 = (1 - _lambda)(
        #     accu_discount_reward_pred[:-1] + discount_value[:-1]
        # )  # (14,1225)

        # last = [
        #     _lambda ** (i - 1) * (accu_discount_reward_pred[i] + discount_value[i])
        #     for i in range(self._c.horizon)
        # ]  # to make horizon -1 lambda returns # # (15,1225)
        # # last = tf.stack(last, 0)  # (15,1225)

        # type3s = [
        #     tf.reduce_sum(type2[:i], axis=0) + last[i] for i in range(self._c.horizon)
        # ]  # (15,1225)

        # revised
        type2_list = []
        discount_reward = reward_pred * pcont
        discount_value = value * pcont
        for k in range(self._c.horizon - 1):  # 14, the k is n-1 in paper
            sum_to_k_discount_reward = tf.reduce_sum(
                discount_reward[: k + 1], 0, keepdims=True
            )  # 1,500. +1 actualy take to k,avoid dimension vanish
            to_kp1_value = tf.expand_dims(
                discount_value[k + 1], 0
            )  # 1,500, the value is next value
            type2 = sum_to_k_discount_reward + to_kp1_value  # 1,500
            type2 = type2 * _lambda ** (k) * (1 - _lambda)
            type2_list.append(type2)  # if conver to array, (horizon-1, 500)

        sum_to_k_discount_reward = tf.reduce_sum(
            discount_reward, 0, keepdims=True
        )  # 1,500
        to_kp1_value = tf.expand_dims(discount_value[-1], 0)  # 1,500
        last_type2 = sum_to_k_discount_reward + to_kp1_value  # 1,500
        last_type2 = last_type2 * _lambda ** (self._c.horizon - 1)  # 1, 500

        type2_list.append(last_type2)

        type3s = tf.concat(type2_list, 0)  # (horizon, 500)

        return type3s

    def update_dreaming(self, obs, acts, obp1s, rewards, dones, record_discounts):
        """
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
        obs = tf.cast(obs, tf.float32)  # after preprocess
        obp1s = tf.cast(obp1s, tf.float32)  # after preprocess
        actions = tf.cast(acts, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        record_discounts = tf.cast(record_discounts, tf.float32)

        with tf.GradientTape() as world_tape:
            """
            the world model, which is _dynamics(RSSM)
            """
            embed = self.encoder(obp1s)  # (25, 50, 1024)
            post, prior = self.dynamics.observe(
                embed, actions
            )  # world model try to dream from first step to last step. each value of post dict is (25, batch_length,30)
            feat = self.dynamics.get_feat(post)  # feat: (25, batch_length, 230)
            # print("world_model_feat:",feat.shape)
            image_pred = self.decoder(
                feat
            )  # image_pred.sample(): (25, batch_length, 64, 64, 3)
            reward_pred = self.reward_decoder(
                feat
            )  # reward_pred.sample(): (25, batch_length)

            likes = {}
            likes["images_prob"] = tf.reduce_mean(image_pred.log_prob(obp1s))
            # print("rewards")
            likes["rewards_prob"] = tf.reduce_mean(reward_pred.log_prob(rewards))
            if (
                self._c.pcont
            ):  # for my aspect, this will make model to learn which step to focus by itself.
                pcont_pred = self._pcont(feat)
                pcont_target = (
                    self._c.discount * record_discounts
                )  # all 1* discount except the done which will be 0. Explicitly make it to learn what is done state.
                likes["pcont_prob"] = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
                likes["pcont_prob"] *= self._c.pcont_scale

            prior_dist = self.dynamics.get_dist(prior)
            post_dist = self.dynamics.get_dist(post)

            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, self._c.free_nats)

            """
            the model loss is exactly the VAE loss of world model(which is VAE sample generator)
            """
            world_loss = self._c.kl_scale * div - sum(
                likes.values()
            )  # like.value contains log prob of image and reward

        world_var = []
        world_var.extend(self.encoder.trainable_variables)
        world_var.extend(self.decoder.trainable_variables)
        world_var.extend(self.dynamics.trainable_variables)
        world_grads = world_tape.gradient(world_loss, world_var)

        with tf.GradientTape() as actor_tape:
            # post:
            imag_feat = self.imagine_ahead(
                post
            )  # scaning to get prior for each prev state, step(policy&world model) for horizon(15) steps.
            # print("imag_feat.shape:",imag_feat.shape) # (500, 15, 230)
            pcont = self._pcont(imag_feat).mean()
            # print("pcont.shape:",pcont.shape) # 500, 15
            lambda_returns = self.lambda_returns(
                imag_feat, pcont, _lambda=self._c.disclam
            )  # an exponentially-weighted average of the estimates V for different k to balance bias and variance
            # print("lambda_returns: ",lambda_returns.shape) # ( )

            actor_loss = -tf.reduce_sum(
                lambda_returns, 0
            )  # !!!!! not using policy gradient !!!! directy maximize return (horizon, batch_size*batch_length)
            actor_loss = tf.reduce_mean(actor_loss)

        actor_var = []
        actor_var.extend(self.actor.trainable_variables)
        actor_grads = actor_tape.gradient(actor_loss, actor_var)

        with tf.GradientTape() as critic_tape:

            discount = tf.stop_gradient(
                tf.math.cumprod(
                    tf.concat([tf.ones_like(pcont[:, :1]), pcont[:, :-2]], 1), 1
                )
            )  # not to effect the world model
            # print("discount:",discount.shape)
            value_pred = self.critic(imag_feat)[:, :-1]
            # print("value_pred:",value_pred.mean().shape)

            target = tf.stop_gradient(tf.transpose(lambda_returns[1:], [1, 0]))
            # print("target:", target.shape)  # (14, 1225)
            # print(
            #     "value_pred.log_prob(target).shape:", value_pred.log_prob(target).shape
            # )  # (14, 1225)
            critic_loss = -tf.reduce_mean(
                discount * value_pred.log_prob(target)
            )  # to directy predict return. gradient is not effecting world model

        critic_var = []
        critic_var.extend(self.critic.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, critic_var)

        self.world_optimizer.apply_gradients(zip(world_grads, world_var))
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_var))
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_var))
        if self.total_step % 1000 == 0:
            self.dynamics.save_weights(self.Dreamer_dynamics_path)
            self.encoder.save_weights(self.Dreamer_encoder_path)
            self.reward_decoder.save_weights(self.Dreamer_reward_decoder_path)
            self.critic.save_weights(self.Dreamer_critic_path)
            self.actor.save_weights(self.Dreamer_actor_path)

        self.total_step += 1
        tf.summary.experimental.set_step(self.total_step)
        print("update finish, save summary...")
        if self.total_step % 20 == 0:
            with self._writer.as_default():
                tf.summary.scalar(
                    "actor_loss/reverse_lambda_return", actor_loss, step=self.total_step
                )
                tf.summary.scalar("critic1_loss", critic_loss, step=self.total_step)
                tf.summary.scalar("world_loss", world_loss, step=self.total_step)
                # tf.summary.scalar(
                #     "average_reward", tf.reduce_mean(rewards), step=self.total_step
                # )

        if self.total_step % 500 == 0:
            print("do image summaries saving!!")
            self._image_summaries(
                {"obs": obs, "actions": actions, "obp1s": obp1s},
                embed,
                image_pred,
            )

    def _image_summaries(self, data, embed, image_pred):
        # print("data['obs']:", data["obs"].shape) #  (50, 10, 64, 64, 1)

        truth = data["obp1s"][:6] + 0.5
        recon = image_pred.mode()[:6]
        init, _ = self.dynamics.observe(
            embed[:6, :5], data["actions"][:6, :5]
        )  # scaning st-1 to get post of st
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self.dynamics.imagine(data["actions"][:6, 5:], init)  # scaning
        openl = self.decoder(self.dynamics.get_feat(prior)).mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = tf.concat([truth, model, error], 2)
        tools.graph_summary(self._writer, tools.video_summary, "agent/openl", openl)

    def save_summary(self, average_reward):
        tf.summary.scalar("average_reward", average_reward, step=self.total_step)
