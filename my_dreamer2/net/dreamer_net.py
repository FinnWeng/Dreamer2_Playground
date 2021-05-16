import tensorflow as tf
import tools
import numpy as np
import cv2
import net.layers as layers
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec
import collections
import utils
import re


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


def world_model_imagine_fresh_action(fn, state_dict, action_array):
    """
    from state, input (action)
    state(stoch): (25*batch_length, 30)
    action_array: (batch_size, horizon,act_dim)
    """
    state_dict_list = []
    feat_list = []
    action_list = []
    current_state = state_dict
    output_dict = {}
    for i in range(action_array.shape[1]):  # to horizon

        current_state, feat, action = fn(current_state, action_array[:, i, :])
        state_dict_list.append(current_state)
        feat_list.append(feat)
        action_list.append(action)
    for k, v in state_dict.items():
        list_of_v = [
            x[k] for x in state_dict_list
        ]  # each is (batch_size*batch_length, 30)
        v_tensor = tf.stack(
            list_of_v, 1
        )  # each is (batch_size*batch_length, horizon, 30)
        output_dict[k] = v_tensor

    feats = tf.stack(feat_list, 1)  # each is (batch_size*batch_length, horizon, 30)

    actions = tf.stack(action_list, 1)  # each is (batch_size*batch_length, horizon, 30)

    return output_dict, feats, actions


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
        self.l1 = tf.keras.layers.Conv2D(2 ** 0 * self.filters, kernel_size=4, **kwargs)
        self.l2 = tf.keras.layers.Conv2D(2 ** 1 * self.filters, kernel_size=4, **kwargs)
        self.l3 = tf.keras.layers.Conv2D(2 ** 2 * self.filters, kernel_size=4, **kwargs)
        self.l4 = tf.keras.layers.Conv2D(2 ** 3 * self.filters, kernel_size=4, **kwargs)
        # self.l5 = tf.keras.layers.Conv2D(8 * self.filters, kernel_size=4, **kwargs)

    def call(self, obs):
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

    def call(self, features):
        kwargs = dict(strides=2, activation=self._act)
        x = self.get("h1", tf.keras.layers.Dense, 32 * self._depth, None)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        x = self.get(
            "h2",
            tf.keras.layers.Conv2DTranspose,
            2 ** (4 - 0 - 2) * self._depth,
            5,
            **kwargs,
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

    def call(self, features):
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
        # (), 3, self._c.units = 400, act=act
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

    def call(self, features):
        x = features
        for index in range(self.layers_num):  # 3
            x = self.get(f"h{index}", tf.keras.layers.Dense, self._units, self._act)(x)
            # print("x:",x.shape) #  (15, 1250, 400)
        mean = self.get(f"hmean", tf.keras.layers.Dense, np.prod(self._shape))(x)
        # print("x:",x.shape)
        mean = tf.reshape(mean, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        # print("x:",x.shape)
        if self._dist == "normal":
            return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))
        if self._dist == "binary":
            return tfd.Independent(tfd.Bernoulli(mean), len(self._shape))
        raise NotImplementedError(self._dist)


class RSSM(tf.keras.Model):
    def __init__(
        self, stoch=30, deter=200, hidden=200, discrete=0, act=tf.nn.elu, cell="gru"
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._activation = act
        self._stoch_size = stoch
        self._deter_size = deter
        self._hidden_size = hidden
        self._discrete = discrete
        if cell == "gru":
            self._cell = tf.keras.layers.GRUCell(
                self._deter_size
            )  # # (output, next_state) = call(input, state)
        elif cell == "gru_layer_norm":
            self._cell = GRUCell(self._deter_size, norm=True)

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, "_modules"):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]

    def initial(self, batch_size):
        # dtype = tf.float32
        # return dict(
        #     mean=tf.zeros([batch_size, self._stoch_size], dtype),
        #     std=tf.zeros([batch_size, self._stoch_size], dtype),
        #     stoch=tf.zeros([batch_size, self._stoch_size], dtype),
        #     deter=self._cell.get_initial_state(None, batch_size, dtype),
        # )  # zero initialization, float32
        dtype = tf.float32
        return dict(
            logit_vector=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
            stoch=tf.zeros([batch_size, self._stoch * self._discrete], dtype),
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

    def get_dist(self, state, stop_grad=False):
        """
        # def get_dist(self, state):
        #     return tfd.MultivariateNormalDiag(state["mean"], state["std"])
        """
        if stop_grad:
            # return tfd.Independent(tfd.OneHotCategorical(tf.stop_gradient(state["logit_vector"])),1)
            return tfd.Independent(
                tools.OneHotDist(tf.stop_gradient(state["logit_vector"])), 1
            )

        else:
            return tfd.Independent(tools.OneHotDist(state["logit_vector"]), 1)

    @tf.function
    def img_step(self, prev_state, prev_action):
        """
        the basic step, run without obs. the one with obs bases on this function
        p(st|st-1,at-1)
        """
        # the image step from St-1 to St (be aware that this is not OBSERVATION)
        # print('prev_state["stoch"]:', prev_state["stoch"].shape)  # (500, 960)
        # print("prev_action:", prev_action.shape)  # (500, 4)
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
        x = self.get("img5", tf.keras.layers.Dense, self._stoch * self._discrete, None)(
            x
        )  # =>(25, 32* _stoch_size)
        logit_vector = tf.reshape(
            x, [-1, self._stoch, self._discrete]
        )  # # =>(25, 32,  _stoch_size)

        """
        Straight-Through Gradients trick is built-in in get_dist 
        """

        stoch = tf.cast(
            self.get_dist({"logit_vector": logit_vector}).sample(), tf.float32
        )
        # print("stoch:", stoch.shape)  # (25, 32 , _stoch_size), one hot vectors

        stoch = tf.keras.layers.Flatten()(stoch)  # # (25, 32*_stoch_size)

        """
        deter is purely from neural network, which is deterministic.
        stoch is sample from distribution decided by deter, which is stochastic.
        """

        prior = {"logit_vector": logit_vector, "stoch": stoch, "deter": deter}
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

        x = self.get("obs3", tf.keras.layers.Dense, self._stoch * self._discrete, None)(
            x
        )
        # print("x:", x.shape)
        logit_vector = tf.reshape(
            x, [-1, self._stoch, self._discrete]
        )  # # =>(25, 32,  _stoch_size)

        """
        Straight-Through Gradients trick is built-in in get_dist 
        """
        stoch = tf.cast(
            self.get_dist({"logit_vector": logit_vector}).sample(), tf.float32
        )  #
        stoch = tf.keras.layers.Flatten()(stoch)  # # (25, 32*_stoch_size)

        post = {"logit_vector": logit_vector, "stoch": stoch, "deter": prior["deter"]}
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
    def __init__(self, env, is_training, step, config):

        # define callback and set networ parameters

        # for discrete space, the space means kinds of choice of action. for continious, it means upper and lower bound of value.
        # so the action_dim means number of action. it is different from for discrete space, kinds of choice.
        self.env = env
        self._c = config
        self._float = prec.global_policy().compute_dtype
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
        self.activation_fn = tf.keras.layers.PReLU
        self.hidden_size = 20

        self.lr = 1e-5
        self.gamma = 0.99
        self.tau = 0.001

        self.eta_x = config.eta_x
        self.eta_r = config.eta_r
        self.eta_gamma = config.eta_gamma
        self.eta_t = config.eta_t
        self.eta_q = config.eta_q

        self.Dreamer_dynamics_path = "./model/Dreamer_dynamics.ckpt"
        self.Dreamer_encoder_path = "./model/Dreamer_encoder.ckpt"
        self.Dreamer_reward_decoder_path = "./model/Dreamer_reward_decoder.ckpt"
        self.Dreamer_critic_path = "./model/Dreamer_critic.ckpt"
        self.Dreamer_actor_path = "./model/Dreamer_actor.ckpt"

        self.dynamics = RSSM(
            self._c.dyn_stoch,
            self._c.dyn_deter,
            self._c.dyn_hidden,
            self._c.dyn_discrete,
            tf.nn.elu,
            self._c.dyn_cell,
        )

        self.encoder = ConvEncoder(self._c.cnn_depth, cnn_actv)
        if self._c.grayscale:
            output_shape = (64, 64, 1)
        else:
            output_shape = (64, 64, 3)

        self.decoder = ConvDecoder(self._c.cnn_depth, cnn_actv, shape=output_shape)

        self.reward_decoder = ValueDecoder((), 4, self._c.units, act=actv)
        if self._c.pcont:
            self._pcont = ValueDecoder((), 4, self._c.units, "binary", act=actv)
            # print("self._pcont:", self._pcont.variables)
        self.critic = ValueDecoder(
            (), 4, self._c.units, act=actv
        )  # critic, no advantage

        if config.slow_value_target or config.slow_actor_target:
            self.slow_critic = ValueDecoder(
                (), 4, self._c.units, act=actv
            )  # critic, no advantage
            self._updates = tf.Variable(0, tf.int64)

        self.actor = ActionDecoder(
            self._actdim,
            4,
            self._c.units,
            self._c.action_dist,
            init_std=self._c.action_init_std,
            act=actv,
        )
        print(
            "tf.zeros_like(acts.low):", tf.zeros_like(actspace.low)
        )  # tf.Tensor([0. 0. 0. 0.], shape=(4,), dtype=float32)
        self.random_actor = tools.OneHotDist(tf.zeros_like(actspace.low)[None])

        model_modules = [self.encoder, self.dynamics, self.decoder, self.reward_decoder]
        self.state = None

        # self.encoder_tffn = tf.function(self.encoder)
        # self.actor_tffn = tf.function(self.actor)
        # self.critic_tffn = tf.function(self.critic)

        # self.world_optimizer = tf.optimizers.Adam(self._c.model_lr)
        # self.actor_optimizer = tf.optimizers.Adam(self._c.actor_lr)
        # self.critic_optimizer = tf.optimizers.Adam(self._c.value_lr)

        self.world_optimizer = tools.Optimizer(
            "world",
            self._c.model_lr,
            self._c.opt_eps,
            self._c.grad_clip,
            self._c.weight_decay,
            self._c.opt,
        )
        self.actor_optimizer = tools.Optimizer(
            "actor",
            self._c.actor_lr,
            self._c.opt_eps,
            self._c.actor_grad_clip,
            self._c.weight_decay,
            self._c.opt,
        )
        self.critic_optimizer = tools.Optimizer(
            "critic",
            self._c.value_lr,
            self._c.opt_eps,
            self._c.value_grad_clip,
            self._c.weight_decay,
            self._c.opt,
        )

        self._writer = tf.summary.create_file_writer(
            "./tf_log", max_queue=1000, flush_millis=20000
        )
        self._step = step
        self.update_step = 1
        # self.update_step = int(15.66*1000)

        self.save_play_img = False

        self.RGB_array_list = []

        # model_list = [self.dynamics,self.encoder,self.reward_decoder, self.critic, self.actor]
        # model_ckpt_path_list = [self.Dreamer_dynamics_path,self.Dreamer_encoder_path,self.Dreamer_reward_decoder_path,self.Dreamer_critic_path, self.Dreamer_actor_path]

    def load_model_weights(self):
        def check_weights_value(list_of_tf_var):
            count = 0.0
            for var in list_of_tf_var:
                var_sum = np.sum(var)
                # print("var.shape:", var.shape)
                # print("var_sum:", var_sum.shape)
                count += var_sum
            return count

        self.dynamics.load_weights(self.Dreamer_dynamics_path)
        # print(
        #     "self.encoder.trainable_variables:",
        #     check_weights_value(self.encoder.trainable_variables),
        # )
        self.encoder.load_weights(self.Dreamer_encoder_path)
        # print(
        #     "self.encoder.trainable_variables:",
        #     check_weights_value(self.encoder.trainable_variables),
        # )
        # self.reward_decoder.load_weights(self.Dreamer_reward_decoder_path)
        # self.critic.load_weights(self.Dreamer_critic_path)

        # print(
        #     "self.actor.trainable_variables:",
        #     check_weights_value(self.actor.trainable_variables),
        # )
        self.actor.load_weights(self.Dreamer_actor_path)
        # print(
        #     "self.actor.trainable_variables:",
        #     check_weights_value(self.actor.trainable_variables),
        # )

        # self.Dreamer_dynamics_path = "./model/Dreamer_dynamics.ckpt"
        # self.Dreamer_encoder_path = "./model/Dreamer_encoder.ckpt"
        # self.Dreamer_reward_decoder_path = "./model/Dreamer_reward_decoder.ckpt"
        # self.Dreamer_critic_path = "./model/Dreamer_critic.ckpt"
        # self.Dreamer_actor_path = "./model/Dreamer_actor.ckpt"

    # if not is_training:

    def reset(self):
        # this reset the state saved for RSSM forwarding
        self.state = None

    def _exploration(self, action, training):
        amount = self._c.expl_amount if training else self._c.eval_noise
        if amount == 0:
            return action
        amount = tf.cast(amount, self._float)
        if "onehot" in self._c.action_dist:
            probs = amount / self._c.num_actions + (1 - amount) * action
            # print("the noisy prob is:", probs)
            return tools.OneHotDist(probs=probs).sample()
        else:
            return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
        raise NotImplementedError(self._c.action_noise)

    def policy(self, obs, training=False):
        # this combine the encoder and actor and get feat to get a totoal agent policy
        if self.state is None:
            obs_len = len(obs["obp1s"])
            latent = self.dynamics.initial(obs_len)
            action = tf.zeros((obs_len, self._actdim), tf.float32)
        else:
            latent, action = self.state

        obs = utils.preprocess(obs, self._c)

        embed = self.encoder(obs["obp1s"])
        latent, _ = self.dynamics.obs_step(
            latent, action, embed
        )  # using post, which using future state to get action # need to to slicing since here should using scan
        feat = self.dynamics.get_feat(latent)
        if training:
            action = self.actor(feat).sample()
        else:
            action = self.actor(feat).mode()

        action = self._exploration(action, training)

        # if self._c.is_discrete:
        #     action = tf.nn.softmax(action, axis=-1)

        state = (latent, action)
        return action, state

    def random_policy(self):
        action = self.random_actor.sample()

        return action

    def imagine_ahead(self, start_state):
        # each value of start_state(post) dict is (25, batch_length,30)
        state = start_state
        # this is different from function imagine. this use fresh action from policy to get next state, and next state.
        flatten = lambda x: tf.reshape(
            x, [-1] + list(x.shape[2:])
        )  # to (batch_size*batch_length,230)
        start = {k: flatten(v) for k, v in state.items()}  # flatten evey entry of dict

        def img_step_fresh_action(prev, _):
            feat = self.dynamics.get_feat(prev)
            inp = tf.stop_gradient(feat) if self._c.behavior_stop_grad else feat
            action = self.actor(inp).sample()
            succ = self.dynamics.img_step(
                prev, action
            )  # the _ is where the storage action is been put
            return succ, feat, action

        control_array = tf.range(
            self._c.horizon
        )  # now ths action is only for control iteration times. So I called it control array
        control_array = tf.reshape(control_array, [1, -1, 1])
        states, imag_feat, imag_action = world_model_imagine_fresh_action(
            img_step_fresh_action, start, control_array
        )  # each is (batch_length*batch_length, horizon, 30)

        # print("imag_feat:",imag_feat.shape)# concate state and obs # (1225,15,  230)

        # start_state: {'logit_vector': <tf.Tensor: shape=(50, 49, 32, 32)
        # states" 'logit_vector': <tf.Tensor: shape=(2450, 15, 32, 32)

        # print('start["logit_vector"]',start["logit_vector"].shape) # (2450, 32, 32)
        # print('start["stoch"]',start["stoch"].shape) # (2450, 1024)
        # print('start["deter"]',start["deter"].shape) # (2450, 800)
        # print("states['logit_vector']",states['logit_vector'].shape) # (2450, 15, 32, 32)
        # print("states['stoch']",states['stoch'].shape) # (2450, 15, 1024)
        # print("states['deter']",states['deter'].shape) # (2450, 15, 800)
        imag_states = {
            k: tf.concat([tf.expand_dims(start[k], 1), v[:, :-1]], 1)
            for k, v in states.items()
        }

        return imag_feat, imag_states, imag_action

    def official_lambda_return(self, reward, value, pcont, bootstrap, lambda_, axis):
        # Setting lambda=1 gives a discounted Monte Carlo return.
        # Setting lambda=0 gives a fixed 1-step return.
        assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
        if isinstance(pcont, (int, float)):
            pcont = pcont * tf.ones_like(reward)
        dims = list(range(reward.shape.ndims))
        dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
        if axis != 0:
            reward = tf.transpose(reward, dims)
            value = tf.transpose(value, dims)
            pcont = tf.transpose(pcont, dims)
        if bootstrap is None:
            bootstrap = tf.zeros_like(value[-1])
        next_values = tf.concat([value[1:], bootstrap[None]], 0)
        inputs = reward + pcont * next_values * (1 - lambda_)
        returns = tools.static_scan(
            lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
            (inputs, pcont),
            bootstrap,
            reverse=True,
        )
        if axis != 0:
            returns = tf.transpose(returns, dims)
        return returns

    def lambda_returns(self, img_feat, pcont, _lambda, actor_ent, state_ent):
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

        if self._c.slow_value_target == True:
            value = self.slow_critic(img_feat).mode()  # (25*batch_length,horizon)
        else:
            value = self.critic(img_feat).mode()  # (25*batch_length,horizon)

        """
        value: (2450, 15)
        reward_pred: (2450, 15)
        pcont: (2450, 15)
        """

        value = tf.transpose(value, [1, 0])
        reward_pred = tf.transpose(reward_pred, [1, 0])
        pcont = tf.transpose(pcont, [1, 0])

        """
        value: (15, 2450)
        reward_pred: (15, 2450)
        pcont: (15, 2450)
        """

        # # revised
        # type2_sum = tf.zeros_like(value[:1])  # 1, 500
        # type3_list = []
        # discount_reward = reward_pred * pcont
        # discount_value = value * pcont
        # for h in range(self._c.horizon - 1):  # 14, the k is n-1 in paper
        #     sum_to_k_to_k_discount_reward = [tf.reduce_sum(
        #         discount_reward[:h]*, 0, keepdims=True
        #     ) for k in range(h)]  # 1,500. +1
        #     sum_to_k_to_k_discount_reward = tf.reduce_sum(tf.stack(sum_to_k_to_k_discount_reward,0),0)

        #     to_kp_value = tf.reduce_sum(
        #         discount_value[h], 0, keepdims=True
        #     )  # (H-h,500) => (1, 500), the value is next value

        #     to_kp1_value = tf.reduce_sum(
        #         discount_value[h + 1], 0, keepdims=True
        #     )  # (H-h,500) => (1, 500), the value is next value

        #     type3 = (1 - _lambda) * (
        #         sum_to_k_discount_reward * (_lambda ** (h)) + type2_sum
        #     ) + (_lambda ** (h + 1)) * to_kp1_value
        #     type3_list.append(type3)

        #     type2_sum = _lambda * (type2_sum + sum_to_k_discount_reward) + to_kp_value

        # # type2_list.append(last_type2)

        # type3s = tf.concat(type3_list, 0)  # (horizon, 500)

        if self._c.future_entropy and tf.greater(self._c.actor_entropy(), 0):
            reward_pred += self._config.actor_entropy() * actor_ent
        if self._c.future_entropy and tf.greater(self._c.actor_state_entropy(), 0):
            reward_pred += self._c.actor_state_entropy() * state_ent

        offical_LamR_result = self.official_lambda_return(
            reward_pred[:-1],
            value[:-1],
            pcont[:-1],
            bootstrap=value[-1],
            lambda_=_lambda,
            axis=0,
        )
        # print("offical_LamR_result:",offical_LamR_result.shape) # (14, 2450)
        # print("type3s:",type3s.shape) # (14, 2450)

        # import pdb

        # pdb.set_trace()

        # return type3s

        return offical_LamR_result

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

        kl_balance = tools.schedule(self._c.kl_balance, self._step)
        kl_free = tools.schedule(self._c.kl_free, self._step)
        kl_scale = tools.schedule(self._c.kl_scale, self._step)

        with tf.GradientTape() as world_tape:
            """
            the world model, which is _dynamics(RSSM)
            """
            embed = self.encoder(obp1s)  # (25, 50, 1024)
            """
            Below are for make "post", which is posterior of VAE/VQVAE
            Could it just come from self.encoder(obp1s) + action???
            """
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
            # print("obp1s:", obp1s.shape)
            # print(
            #     "image_pred.log_prob(obp1s):", image_pred.log_prob(obp1s).shape
            # )  # 50, 10
            # print(
            #     "image_pred:", image_pred.batch_shape, image_pred.event_shape
            # )  # (50, 10) (64, 64, 3)
            likes["images_prob"] = -self.eta_x * tf.reduce_mean(
                image_pred.log_prob(obp1s)
            )
            # print("rewards")
            likes["rewards_prob"] = -self.eta_r * tf.reduce_mean(
                reward_pred.log_prob(rewards)
            )
            if (
                self._c.pcont
            ):  # for my aspect, this will make model to learn which step to focus by itself.
                pcont_pred = self._pcont(feat)
                pcont_target = (
                    self._c.discount * record_discounts
                )  # all 1* discount except the done which will be 0. Explicitly make it to learn what is done state.
                likes["pcont_prob"] = -tf.reduce_mean(
                    pcont_pred.log_prob(pcont_target)
                )  # shape = (50,10)
                likes["pcont_prob"] *= self._c.pcont_scale

            prior_dist = self.dynamics.get_dist(prior)
            stop_prior_dist = self.dynamics.get_dist(prior, stop_grad=True)

            post_dist = self.dynamics.get_dist(post)
            stop_post_dist = self.dynamics.get_dist(post, stop_grad=True)

            # div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            # div = tf.maximum(div, self._c.free_nats)

            mix = 1 - kl_balance

            value_lhs = value = tfd.kl_divergence(post_dist, stop_prior_dist)
            value_rhs = tfd.kl_divergence(stop_post_dist, prior_dist)
            loss_lhs = tf.maximum(tf.reduce_mean(value_lhs), kl_free)
            loss_rhs = tf.maximum(tf.reduce_mean(value_rhs), kl_free)
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
            loss *= kl_scale

            world_loss = loss + sum(likes.values())

            """
            the model loss is exactly the VAE loss of world model(which is VAE sample generator)
            """

            # world_loss = self._c.kl_scale * kl_blancing_div - sum(
            #     likes.values()
            # )  # like.value contains log prob of image and reward

        # world_var = []
        # world_var.extend(self.encoder.trainable_variables)
        # world_var.extend(self.decoder.trainable_variables)
        # world_var.extend(self.dynamics.trainable_variables)
        # world_grads = world_tape.gradient(world_loss, world_var)
        world_model_parts = [
            self.encoder,
            self.decoder,
            self.dynamics,
            self.reward_decoder,
            self._pcont,
        ]
        self.world_optimizer(world_tape, world_loss, world_model_parts)

        self._update_slow_target()

        with tf.GradientTape() as actor_tape:

            drop_last = lambda x: x[:, :-1, :]
            post = {k: drop_last(v) for k, v in post.items()}

            # post:
            imag_feat, imag_state, imag_action = self.imagine_ahead(
                post
            )  # scaning to get prior for each prev state, step(policy&world model) for horizon(15) steps.

            discount_inp = self.dynamics.get_feat(
                imag_state
            )  # since the feat s prev feat
            discount = self._pcont(discount_inp).mean()
            weights = tf.stop_gradient(
                tf.math.cumprod(
                    tf.concat([tf.ones_like(discount[:, :1]), discount[:, :-2]], 1), 1
                )
            )  # not to effect the world model
            # print("weights:",weights.shape) # (2450, 14)

            actor_inp = (
                tf.stop_gradient(imag_feat) if self._c.behavior_stop_grad else imag_feat
            )
            policy = self.actor(actor_inp)

            actor_ent = policy.entropy()  # (2450, 15)
            state_ent = self.dynamics.get_dist(imag_state).entropy()  # (2450, 15, 32)

            imag_action_prob = policy.log_prob(imag_action)
            # print("imag_action_prob:", imag_action_prob) # (2450, 15)

            lambda_returns = self.lambda_returns(
                imag_feat,
                discount,
                _lambda=self._c.discount_lambda,
                actor_ent=actor_ent,
                state_ent=state_ent,
            )  # an exponentially-weighted average of the estimates V for different k to balance bias and variance

            value_diff = tf.stop_gradient(
                tf.transpose(lambda_returns, [1, 0])
                - self.critic(imag_feat[:, :-1]).mode()
            )

            # print("value_diff:", value_diff)

            actor_target = imag_action_prob[:, :-1] * value_diff

            # print("lambda_returns: ", lambda_returns.shape)  # (14, 2450)
            # print("actor_loss:",actor_loss.shape) # (2450, 14)
            # print("discount:",discount.shape) # (2450, 14)

            actor_mix = self._c.imag_gradient_mix()

            mix_actor_target = (
                tf.transpose(lambda_returns, [1, 0]) * actor_mix
                + (1 - actor_mix) * actor_target
            )
            # print("mix_actor_loss:",mix_actor_loss) # (2450, 14)

            if not self._c.future_entropy and tf.greater(self._c.actor_entropy(), 0):
                mix_actor_target += self._c.actor_entropy() * actor_ent[:, :-1]
            if not self._c.future_entropy and tf.greater(
                self._c.actor_state_entropy(), 0
            ):
                mix_actor_target += self._c.actor_state_entropy() * state_ent[:-1]

            actor_loss = -tf.reduce_mean(weights * mix_actor_target)

        # actor_var = []
        # actor_var.extend(self.actor.trainable_variables)
        # actor_grads = actor_tape.gradient(actor_loss, actor_var)
        actor_model_parts = [self.actor]

        with tf.GradientTape() as critic_tape:

            value_pred = self.critic(imag_feat)[:, :-1]
            # print("value_pred:",value_pred.mean().shape)

            target = tf.stop_gradient(tf.transpose(lambda_returns, [1, 0]))
            # print("target:", target.shape)  # (14, 1225)
            # print(
            #     "value_pred.log_prob(target).shape:", value_pred.log_prob(target).shape
            # )  # (14, 1225)
            critic_loss = tf.reduce_mean(
                weights * -value_pred.log_prob(target)
            )  # to directy predict return. gradient is not effecting world model

        critic_model_parts = [self.critic]

        self.actor_optimizer(actor_tape, actor_loss, actor_model_parts)
        self.critic_optimizer(critic_tape, critic_loss, critic_model_parts)

        if self.update_step % 2000 == 0:
            self.dynamics.save_weights(self.Dreamer_dynamics_path)
            self.encoder.save_weights(self.Dreamer_encoder_path)
            self.reward_decoder.save_weights(self.Dreamer_reward_decoder_path)
            self.critic.save_weights(self.Dreamer_critic_path)
            self.actor.save_weights(self.Dreamer_actor_path)

        self.update_step += 1
        tf.summary.experimental.set_step(self.update_step)

        if self.update_step % 20 == 0:
            print("update finish, save summary..., now update step:", self.update_step)
            with self._writer.as_default():
                tf.summary.scalar(
                    "actor_loss",
                    actor_loss,
                    step=self._step.numpy() * self._c.action_repeat,
                )
                tf.summary.scalar(
                    "actor_ent",
                    tf.reduce_mean(actor_ent),
                    step=self._step.numpy() * self._c.action_repeat,
                )
                tf.summary.scalar(
                    "imag_gradient_mix",
                    actor_mix,
                    step=self._step.numpy() * self._c.action_repeat,
                )

                tf.summary.scalar(
                    "critic1_loss",
                    critic_loss,
                    step=self._step.numpy() * self._c.action_repeat,
                )
                tf.summary.scalar(
                    "world_loss",
                    world_loss,
                    step=self._step.numpy() * self._c.action_repeat,
                )

        if self.update_step % 2000 == 0:
            print("do image summaries saving!!")
            self._image_summaries(
                {"obs": obs, "actions": actions, "obp1s": obp1s},
                embed,
                image_pred,
            )
        # print("self.dynamics:", self.dynamics.summary())

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
        tf.summary.scalar(
            "average_reward/train_return", average_reward, step=self.update_step
        )

    def _update_slow_target(self):
        if self._c.slow_value_target or self._c.slow_actor_target:
            if self._updates % self._c.slow_target_update == 0:
                mix = self._c.slow_target_fraction
                for s, d in zip(self.critic.variables, self.slow_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)


class GRUCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = tf.keras.layers.Dense(
            3 * size, use_bias=norm is not None, **kwargs
        )
        if norm:
            self._norm = tf.keras.layers.LayerNormalization(dtype=tf.float32)

    @property
    def state_size(self):
        return self._size

    def call(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))
        if self._norm:
            dtype = parts.dtype
            parts = tf.cast(parts, tf.float32)
            parts = self._norm(parts)
            parts = tf.cast(parts, dtype)
        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update = tf.nn.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]
