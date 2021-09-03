from tensorflow_probability.python.distributions import Distribution, Categorical
from tensorflow_probability.python.internal import reparameterization, samplers, parameter_properties
import tensorflow as tf
import six


class MultiCategorical(Distribution):
    """Categorical distribution where the choice from each category are related"""

    def __init__(self, logits, n_categories: int):
        """

        Parameters
        ----------
        logits:
            NDMatrix where each entry is an unnormalized log-probability for a certain category.
            Shape have at least as many dimensions as the number of categories.
            Outer dimensions are treated as batch dimensions
        n_categories: int
            Number of categories
        """
        super().__init__(
            dtype=tf.int32, reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            validate_args=False, allow_nan_stats=True
        )
        self.input_shape = tf.shape(logits)
        self._outer_shape = self.input_shape[:-n_categories]
        self.batch_dims = tf.shape(self.input_shape)[0] - n_categories
        self.batch_size = tf.reduce_prod(self._outer_shape)
        self.category_shape = self.input_shape[-n_categories:]
        self.n_categories = n_categories
        self._dtype = self.input_shape.dtype

        # Flatten to seem like a single set of categories for each batch
        self._logits = logits
        self._logits_flat = tf.reshape(self._logits, (self.batch_size, -1))
        self._probs_flat = tf.nn.softmax(self._logits_flat)
        self._probs = tf.reshape(self._probs_flat, self.input_shape)

    def _sample_n(self, n, seed=None, **kwargs):
        # Make the sampling
        # From TFP/categorical.py -> TODO(b/147874898): Remove workaround for seed-sensitive tests.
        if seed is None or isinstance(seed, six.integer_types):
            idx_samples = tf.random.categorical(self._logits_flat, n, seed=seed, dtype=self._dtype)
        else:
            idx_samples = samplers.categorical(self._logits_flat, n, seed=seed, dtype=self._dtype)

        # Map the index in the array to the multi-categorical shape
        idx_samples = tf.reshape(idx_samples, [-1])  # (batch_size, n) -> (batch_size * n)
        ind_samples = tf.unravel_index(
            idx_samples,
            self.category_shape
        )  # (n_dim, self.n_samples), Samples are ordered (n samples from batch_0, n samples from batch_1, ...)
        #  So, the last varying dimension is "n"
        ind_samples = tf.reshape(ind_samples, tf.concat([[self.n_categories], self.batch_shape, [n]], 0))

        # TFP expects the number of samples to be the first dimension
        perm = tf.concat([
            [self.batch_dims + 1], tf.range(1, self.batch_dims + 1), [0]
        ], 0)
        return tf.transpose(ind_samples, perm=perm)  # (n, batch_size, n_dim)

    def _event_shape(self):
        return tf.TensorShape([self.n_categories])

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

    @property
    def probs(self):
        return self._probs

    def _log_prob(self, value):
        return tf.gather_nd(self._logits, value, batch_dims=self.batch_dims)

    def _batch_shape(self):
        return self._outer_shape

    def _batch_shape_tensor(self):
        return tf.TensorSpec(self._outer_shape)

    def _entropy(self, **kwargs):
        entropy = Categorical(logits=self._logits_flat).entropy()  # (batch_size,)
        return tf.reshape(entropy, self._outer_shape)

    def _mode(self, **kwargs):
        # Get the most-probable sample for each
        idx_max = tf.argmax(self._probs_flat, axis=-1, output_type=self.dtype)  # batch_shape

        # Get the coordinates
        ind_mode = tf.unravel_index(
            idx_max,
            self.category_shape
        )  # (n_dim, batch_size)

        # Convert them to the proper shape
        ind_mode = tf.transpose(ind_mode)  # (batch_size, n_dim)
        return tf.reshape(ind_mode, tf.concat((self._outer_shape, [self.n_categories]), axis=0))
