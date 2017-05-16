"""
The :mod:`svd_ops` module implements several low-rank decomposition algorithms, including `Simon Funk's
<http://sifter.org/~simon/journal/20061211.html>`_ *SVD* algorithm and `Yehuda Koren's
<http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf>`_
*Asymmetric-SVD* and *SVD++* algorithms.

For the rest of the code, let :math:`r_{ui}` be the rating user :math:`u` gives to item :math:`i`.
Let :math:`\\hat{r}_{ui}` be the predicted rating of user :math:`u` to item :math:`i` and finally let:

.. math::
    e_{ui} = r_{ui} - \hat{r}_{ui}.
"""

import tensorflow as tf


def predict_svd(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    """
    This algorithm predicts :math:`\\hat{r}_{ui}` by:

    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u

    where
    * :math:`b_u, b_i` are biases for user :math:`u` and item :math:`i` respectively,
    * :math:`q_i, p_u` are vectors of latent factors for user :math:`u` and item :math:`i` respectively.
    
    :param int user_batch: The user ids of the users.
    :param int item_batch: The item ids of the items.
    :param int user_num: How many users in the whole dataset.
    :param int item_num: How many items in the whole dataset.
    :param int dim: Number of latent factors to compute with.
    :param str device: Which device to use for the computations.
    """
    # Find global variables first.
    with tf.device("/cpu:0"):
        # Tensors for individual user and item biases.
        with tf.variable_scope("biases"):
            user_bias_tensor = tf.get_variable("user_bias_tensor", shape=[user_num])
            item_bias_tensor = tf.get_variable("item_bias_tensor", shape=[item_num])

        # Tensors for the latent factors of the users and items.
        with tf.variable_scope("latent_factors"):
            user_latent_tensor = tf.get_variable("user_latent_tensor", shape=[user_num, dim],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            item_latent_tensor = tf.get_variable("item_latent_tensor", shape=[item_num, dim],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.02))

        # The global mean of the data set.
        with tf.variable_scope("mean"):
            global_mean = tf.get_variable("global_mean", shape=[])

        # The biases of the users and items in this batch.
        bu = tf.nn.embedding_lookup(user_bias_tensor, user_batch, name="bu")
        bi = tf.nn.embedding_lookup(item_bias_tensor, item_batch, name="bi")

        # The latent factors of the users and items in this batch.
        pu = tf.nn.embedding_lookup(user_latent_tensor, user_batch, name="pu")
        qi = tf.nn.embedding_lookup(item_latent_tensor, item_batch, name="qi")

    # Calculate a prediction.
    with tf.device(device):
        predicted = bu + bi + tf.reduce_sum(pu * qi, 1)
        predicted = tf.add(predicted, global_mean, name="predicted_value")
        bias_loss = tf.nn.l2_loss(bu) + tf.nn.l2_loss(bi)
        latent_loss = tf.nn.l2_loss(pu) + tf.nn.l2_loss(qi)
        regularizer = tf.add(bias_loss, latent_loss, name="regularizer")

    return predicted, regularizer


def optimize_svd(predicted, regularizer, actual_batch, learning_rate=0.001, reg=0.1, device="/cpu:0"):
    """
    Trains using the ``train_set`` by minimizing the following error:

    .. math::
        \sum_{r_{ui} \in \R_{train}} \left(r_{ui} - \hat{r}_{ui} \\right)^2 +
        \lambda\\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\\right)
    :param predicted: A tensor containing the predicted ratings.
    :param regularizer: A tensor containing the part of the error.
    :param actual_batch: The ratings of the users to the items.
    :param learning_rate: Learning rate.
    :param reg: Regularizer.
    :param device: Which device to use for the computations.
    :return: 
    """
    global_step = tf.train.get_global_step()
    assert global_step is not None

    with tf.device(device):
        reg = tf.constant(reg, dtype=tf.float32, name="l2_reg")
        cost = tf.nn.l2_loss(predicted - actual_batch)
        cost = tf.add(cost, tf.multiply(regularizer, reg), name="cost")
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    return cost, train_op
