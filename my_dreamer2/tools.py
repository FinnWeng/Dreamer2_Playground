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


class OneHotDist:
    def __init__(self, logits=None, probs=None):
        self._dist = tfd.Categorical(logits=logits, probs=probs)
        self._num_classes = self.mean().shape[-1]
        self._dtype = prec.global_policy().compute_dtype

    @property
    def name(self):
        return "OneHotDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.prob(indices)

    def log_prob(self, events):
        indices = tf.argmax(events, axis=-1)
        return self._dist.log_prob(indices)

    def mean(self):
        return self._dist.probs_parameter()

    def mode(self):
        return self._one_hot(self._dist.mode())

    def sample(self, amount=None):
        amount = [amount] if amount else []
        indices = self._dist.sample(*amount)
        sample = self._one_hot(indices)
        probs = self._dist.probs_parameter()
        sample += tf.cast(probs - tf.stop_gradient(probs), self._dtype)
        return sample

    def _one_hot(self, indices):
        return tf.one_hot(indices, self._num_classes, dtype=self._dtype)
