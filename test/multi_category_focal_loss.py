def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
    当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-6
    alpha = tf.constant(alpha, dtype=tf.float32)
    # alpha = tf.constant([[1],[1],[1],[1],[1]], dtype=tf.float32)
    # alpha = tf.constant_initializer(alpha)
    gamma = float(gamma)

    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss1_fixed


focal_loss = multi_category_focal_loss1([0.1, 0.2, 0.3])
loss = focal_loss([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
print(loss)
