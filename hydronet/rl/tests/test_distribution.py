import tensorflow as tf

from hydronet.rl.tf.distribution import MultiCategorical


def test_multicategorical():
    # Test with a single batch to make sure actions work
    dist = MultiCategorical(tf.constant([
        [[0.2, 0.1], [0.3, 0.4]]
    ]))
    assert dist.sample(8).shape == (8, 1, 2)
    assert dist.mode().shape == (1, 2)
    assert (dist.mode() == [1, 1]).numpy().all()
    assert dist.prob([[0, 1]]) == 0.1

    # Use two batches to make sure samples are the same
    dist = MultiCategorical(tf.constant([
        [[0.1, 0.9], [0, 0]],  # Only [0, 0] and [0, 1]
        [[0, 0], [0.1, 0.9]]   # Only [1, 0] and [1, 1]
    ]))
    samples = dist.sample(3)
    assert samples.shape == (3, 2, 2)

    #  Testing the results
    samples = samples.numpy()
    assert samples[:, 0, 0].max() == 0
    assert samples[:, 0, 1].max() == 1

    assert (dist.mode().numpy() == [[0, 1], [1, 1]]).all()
    assert (dist.prob([[0, 0], [1, 1]]) == [0.1, 0.9]).numpy().all()
