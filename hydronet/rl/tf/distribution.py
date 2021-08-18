from tensorflow_probability.python.distributions import Distribution, Normal
from tensorflow_probability.python.internal import reparameterization, samplers, parameter_properties
import tensorflow as tf
import six


class MultiCategorical(Distribution):
    """Categorical distribution where the choice from each category are related"""

    def __init__(self, probs: tf.Tensor):
        super().__init__(
            dtype=tf.int32, reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            validate_args=False, allow_nan_stats=True
        )
        self._shape = tf.shape(probs)
        self._batch_size = self._shape[0]
        self._dim = len(self._shape) - 1
        self._dtype = self._shape.dtype

        # Flatten to seem like a single set of categories for each batch
        self._logits = tf.reshape(tf.math.log(probs), [self._shape[0], -1])

    def _sample_n(self, n, seed=None, **kwargs):
        # Make the sampling
        # From TFP/categorical.py -> TODO(b/147874898): Remove workaround for seed-sensitive tests.
        if seed is None or isinstance(seed, six.integer_types):
            idx_samples = tf.random.categorical(self._logits, n, seed=seed, dtype=self._dtype)
        else:
            idx_samples = samplers.categorical(self._logits, n, seed=seed, dtype=self._dtype)

        # Map the index in the array to the multi-categorical shape
        idx_samples = tf.reshape(idx_samples, [-1])  # (batch_size, n) -> (batch_size * n)
        ind_samples = tf.unravel_index(
            idx_samples,
            self._shape[1:]
        )  # (n_dim, self.n_samples), Samples are ordered (n samples from batch_0, n samples from batch_1, ...)
        #  So, the last varying dimension is "n"
        ind_samples = tf.reshape(ind_samples, [self._dim, self._batch_size, n])

        # TFP expects the number of samples to be the first dimension
        return tf.transpose(ind_samples, perm=[2, 1, 0])  # (n, batch_size, n_dim)

    def _event_shape(self):
        return tf.TensorShape([self._dim])

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            logits=parameter_properties.ParameterProperties(
                event_ndims=2
            )
        )

    @property
    def logits(self):
        return self._logits

    def _log_prob(self, value):
        return tf.gather_nd(self._logits, value, batch_dims=1)

    def _batch_shape(self):
        return self._batch_size

    def _batch_shape_tensor(self):
        return tf.TensorSpec((self._batch_size,))

    def _mode(self, **kwargs):
        # Get the most-probable sample for each
        idx_max = tf.argmax(self._logits, axis=1, output_type=self.dtype)  # (batch_size,)

        # Get the coordinates
        ind_mode = tf.unravel_index(
            idx_max,
            self._shape[1:]
        )  # (n_dim, batch_size)
        return tf.transpose(ind_mode)
