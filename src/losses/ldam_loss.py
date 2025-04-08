import tensorflow as tf


class LDAMLoss(tf.keras.losses.Loss):
    def __init__(self, cls_num_list, m_max=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()

        m_list = 1./ np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (m_max / np.max(m_list))
        self.m_list = tf.convert_to_tensor(m_list, dtype=tf.float32)

        assert s > 0
        self.s = s
        self.weight = weight
        self.n_class = len(cls_num_list)

    def call(self, y_true, y_pred):
      y_true = tf.cast(y_true, tf.int32)
      y_true = tf.reshape(y_true, [-1])#need to convert to (batch_size,)

      # One-hot encode labels to match y_pred shape
      y_true_one_hot = tf.one_hot(y_true, depth=len(self.m_list), dtype=tf.float32)

      # Compute margin per class
      batch_m = tf.matmul(y_true_one_hot, tf.reshape(self.m_list, (-1, 1)))

      x_m = y_pred - batch_m  # Subtract margin from the correct class

      output = y_true_one_hot * x_m + (1 - y_true_one_hot) * y_pred

      loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true_one_hot, logits=output*self.s)

      return tf.reduce_mean(loss)  # Return mean loss