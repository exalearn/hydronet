import tensorflow as tf

from hydronet.rl.tf.distribution import MultiCategorical


def test_multicategorical():
    # Test with a single batch to make sure actions work
    dist = MultiCategorical(tf.math.log(tf.Variable([
        [[0.2, 0.1], [0.3, 0.4]]
    ])), 2)
    assert dist.sample(8).shape == (8, 1, 2)
    assert dist.mode().shape == (1, 2)
    assert (dist.mode() == [1, 1]).numpy().all()
    assert dist.prob([[0, 1]]) == 0.1

    # Use two batches to make sure samples are the same
    dist = MultiCategorical(tf.math.log(tf.constant([
        [[0.1, 0.9], [0, 0]],  # Only [0, 0] and [0, 1]
        [[0, 0], [0.1, 0.9]]  # Only [1, 0] and [1, 1]
    ])), 2)
    samples = dist.sample(3)
    assert samples.shape == (3, 2, 2)

    #  Testing the results
    samples = samples.numpy()
    assert samples[:, 0, 0].max() == 0
    assert samples[:, 0, 1].max() == 1

    assert (dist.mode().numpy() == [[0, 1], [1, 1]]).all()
    assert (dist.prob([[0, 0], [1, 1]]) == [0.1, 0.9]).numpy().all()

    # Make sure it has derivatives
    input_probs = tf.Variable([
        [[0.1, 0.9], [0, 0]],  # Only [0, 0] and [0, 1]
        [[0, 0], [0.1, 0.9]]  # Only [1, 0] and [1, 1]
    ])
    with tf.GradientTape() as tape:
        dist = MultiCategorical(input_probs, 2)
        probs = dist.log_prob(dist.mode())
        psum = tf.reduce_sum(probs)
    grads = tape.gradient(psum, input_probs)
    assert all(not tf.reduce_all(tf.math.is_nan(g)).numpy() for g in grads)

    # Try with a complex batch shape
    input_probs = tf.Variable([[
        [[[0.1, 0.9], [0, 0]], [[0, 0], [0.1, 0.9]]],
        [[[0, 0], [0.9, 0.1]], [[0, 0], [0.1, 0.9]]],
    ]])
    dist = MultiCategorical(input_probs, 2)
    assert tf.equal(dist.mode(), [[
        [[0, 1], [1, 1]],
        [[1, 0], [1, 1]]
    ]]).numpy().all()

    # Compute the KL divergence with self
    kl = dist.kl_divergence(dist)
    assert kl.shape == (1, 2, 2)

    # Compute entropy
    assert dist.entropy().shape == (1, 2, 2)
    assert dist.entropy().numpy().all() > 0

    # Test derivatives of entropy
    with tf.GradientTape() as tape:
        dist = MultiCategorical(input_probs, 2)
        entropy = tf.reduce_sum(dist.entropy())
    grads = tape.gradient(entropy, input_probs)
    assert not tf.reduce_any(tf.math.is_nan(grads)).numpy()
