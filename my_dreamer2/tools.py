import datetime
import io
import pathlib
import pickle
import re
import uuid

# import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd


def count_episodes(directory):
    filenames = directory.glob("*.npz")
    lengths = [int(n.stem.rsplit("-", 1)[-1]) - 1 for n in filenames]
    episodes, steps = len(lengths), sum(lengths)
    return episodes, steps


def graph_summary(writer, fn, *args):
    step = tf.summary.experimental.get_step()

    def inner(*args):
        tf.summary.experimental.set_step(step)
        with writer.as_default():
            fn(*args)

    return tf.numpy_function(inner, args, [])


# def video_summary(name, video, step=None, fps=20):
#     #   name = name if isinstance(name, str) else name.decode('utf-8')
#     name = str(name)

#     if np.issubdtype(video.dtype, np.floating):
#         video = np.clip(255 * video, 0, 255).astype(np.uint8)
#     print("video.shape:", video.shape)
#     B, T, H, W, C = video.shape
#     try:
#         # frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
#         frames = video
#         summary = tf1.Summary()
#         image = tf1.Summary.Image(height=H, width=W, colorspace=C)
#         image.encoded_image_string = encode_gif(frames, fps)
#         summary.value.add(tag=name + "/gif", image=image)
#         tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
#     except (IOError, OSError) as e:
#         print("GIF summaries require ffmpeg in $PATH.", e)
#         frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
#         tf.summary.image(name + "/grid", frames, step)


def video_summary(name, video, step=None, fps=20):
    # print("name:", name)
    # name = name if isinstance(name, str) else name.decode("utf-8")
    name = str(name)
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    try:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)
        summary.value.add(tag=name + "/gif", image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print("GIF summaries require ffmpeg in $PATH.", e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name + "/grid", frames, step)


def encode_gif(frames, fps):
    from subprocess import Popen, PIPE

    h, w, c = frames[0].shape
    pxfmt = {1: "gray", 3: "rgb24"}[c]
    cmd = " ".join(
        [
            f"ffmpeg -y -f rawvideo -vcodec rawvideo",
            f"-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex",
            f"[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
            f"-r {fps:.02f} -f gif -",
        ]
    )
    proc = Popen(cmd.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
    del proc
    return out


class TanhBijector(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__(
            forward_min_event_ndims=0, validate_args=validate_args, name=name
        )

    def _forward(self, x):
        return tf.nn.tanh(x)

    def _inverse(self, y):
        dtype = y.dtype
        y = tf.cast(y, tf.float32)
        y = tf.where(
            tf.less_equal(tf.abs(y), 1.0),
            tf.clip_by_value(y, -0.99999997, 0.99999997),
            y,
        )
        y = tf.atanh(y)
        y = tf.cast(y, dtype)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return tf.reduce_mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)


class OneHotDist(tfd.OneHotCategorical):
    def __init__(self, logits=None, probs=None, dtype=None):
        self._sample_dtype = dtype or prec.global_policy().compute_dtype
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        return tf.cast(super().mode(), self._sample_dtype)

    def sample(self, sample_shape=(), seed=None):
        # Straight through biased gradient estimator.
        sample = tf.cast(super().sample(sample_shape, seed), self._sample_dtype)
        probs = super().probs_parameter()
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += tf.cast(probs - tf.stop_gradient(probs), self._sample_dtype)
        return sample


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        step = tf.cast(step, tf.float32)
        match = re.match(r"linear\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = tf.clip_by_value(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r"warmup\((.+),(.+)\)", string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = tf.clip_by_value(step / warmup, 0, 1)
            return scale * value
        match = re.match(r"exp\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r"horizon\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = tf.clip_by_value(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def static_scan(fn, inputs, start, reverse=False):
    last = start
    outputs = [[] for _ in tf.nest.flatten(start)]
    indices = range(len(tf.nest.flatten(inputs)[0]))
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = tf.nest.map_structure(lambda x: x[index], inputs)

        last = fn(last, inp)
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [tf.stack(x, 0) for x in outputs]
    return tf.nest.pack_sequence_as(start, outputs)
    # inputs = reward + pcont * next_values * (1 - lambda_)
    # fn = lambda agg, cur: cur[0] + cur[1] * lambda_ * agg
    # i = reverse(index)
    #
    # start = input[i] + pcont[i]* lambda_ * start
    #
    # (1- lambda) * sigma(lambda*Vn) + lambda* Vn+1
    # Vn = sigma(r) + vn
    # (1 - lambda)  (sigma(v1*lambda**1+...+vn*lambda**n) + sigma_to_n(sigma_to_k(r*lambda**n))) + lambda* Vn+1 # two kind of sigma


class Optimizer(tf.Module):
    def __init__(
        self,
        name,
        lr,
        eps=1e-4,
        clip=None,
        wd=NotImplementedError,
        wd_pattern=r".*",
        opt="adam",
    ):
        self._name = name
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: tf.optimizers.Adam(lr, epsilon=eps),
            "nadam": lambda: tf.optimizers.Nadam(lr, epsilon=eps),
            "adamax": lambda: tf.optimizers.Adamax(lr, epsilon=eps),
            "sgd": lambda: tf.optimizers.SGD(lr),
            "momentum": lambda: tf.optimizers.SGD(lr, 0.9),
        }[opt]()

    def __call__(self, tape, loss, modules):
        modules = modules if hasattr(modules, "__len__") else (modules,)
        varibs = tf.nest.flatten([module.variables for module in modules])
        # count = sum(np.prod(x.shape) for x in varibs)
        # print(f"Found {count} {self._name} parameters.")
        grads = tape.gradient(loss, varibs)
        norm = tf.linalg.global_norm(grads)
        if self._clip:
            grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)

        if self._wd:
            self._apply_weight_decay(varibs)

        self._opt.apply_gradients(zip(grads, varibs))

    def _apply_weight_decay(self, varibs):
        """
        The self._wd_pattern is trivial, so actually it doesn't apply any weight decay
        """
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            print("Applied weight decay to variables:")
        for var in varibs:
            if re.search(self._wd_pattern, self._name + "/" + var.name):
                if nontrivial:
                    print("- " + self._name + "/" + var.name)
                var.assign((1 - self._wd) * var)
