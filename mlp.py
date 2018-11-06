import  tensorflow as tf



def single_layer_classifier(h, label, n_way, scope, reuse=False):
    """
    :param h: hidden placeholder
    :param l: label placeholder
    :param n_way: number of class
    :param scope: scope name for this module
    :param reuse:
    :return:
    """
    # Here can reuse=True or reuse=Flase
    with tf.variable_scope(scope, reuse=reuse):
        # [b, 4, 4, 16] => [b, -1]
        h0 = tf.layers.flatten(h)
        # => [b, 10]
        logits = tf.layers.dense(h0, n_way)
        # => [b]
        pred = tf.argmax(logits, 1)

        # TODO: when your label=[5,6,7,8,9]> n_way=5, it wont occur an error!
        tf.assert_less(label, tf.to_float(n_way))
        label = tf.one_hot(tf.to_int32(label), n_way)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label))

    return loss, pred