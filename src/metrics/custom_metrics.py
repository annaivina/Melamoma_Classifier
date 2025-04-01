import tensorflow as tf 


class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="balanced_accuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(tf.cast(y_true, tf.bool))
        y_pred = tf.squeeze(tf.cast(tf.round(y_pred), tf.bool))


        if sample_weight is None:
            sample_weight = tf.ones_like(tf.cast(y_true, tf.float32))

        sample_weight = tf.reshape(tf.cast(sample_weight, tf.float32), tf.shape(y_true))

        true_p = tf.cast(tf.logical_and(y_true, y_pred), tf.float32) * sample_weight  #for some reason graph doesnt like boolen masks.. 
        true_n = tf.cast(tf.logical_and(~y_true, ~y_pred), tf.float32) * sample_weight 
        false_p = tf.cast(tf.logical_and(~y_true, y_pred), tf.float32) * sample_weight 
        false_n = tf.cast(tf.logical_and(y_true, ~y_pred), tf.float32) * sample_weight 


        self.tp.assign_add(tf.reduce_sum(true_p))
        self.tn.assign_add(tf.reduce_sum(true_n))
        self.fp.assign_add(tf.reduce_sum(false_p))
        self.fn.assign_add(tf.reduce_sum(false_n))
    
    def result(self):
        sensitivity = self.tp / (self.tp + self.fn + 1e-7)
        specificity = self.tn / (self.tn + self.fp + 1e-7)
       
        return (sensitivity + specificity) / 2.0
        

    def reset_state(self):
        for var in [self.tp, self.tn, self.fp, self.fn]:
            var.assign(0.0)
    
    def get_config(self):
      config = super().get_config()
      return config




class CustomF1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(CustomF1Score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)


        if sample_weight is None:
            sample_weight = tf.ones_like(tf.cast(y_true, tf.float32))

        sample_weight = tf.reshape(tf.cast(sample_weight, tf.float32), tf.shape(y_true))

        true_p = tf.cast(tf.logical_and(y_true, y_pred), tf.float32) * sample_weight  #for some reason graph doesnt like boolen masks.. 
        false_p = tf.cast(tf.logical_and(~y_true, y_pred), tf.float32) * sample_weight 
        false_n = tf.cast(tf.logical_and(y_true, ~y_pred), tf.float32) * sample_weight 


        self.tp.assign_add(tf.reduce_sum(true_p))
        self.fp.assign_add(tf.reduce_sum(false_p))
        self.fn.assign_add(tf.reduce_sum(false_n))


    def result(self):
        precision = self.tp/(self.tp + self.fp)
        recall =self.tp/(self.tp + self.fn)
        return 2 * (precision * recall) / (precision + recall + 1e-7)  # Avoid division by zero

    def reset_state(self):
        for var in [self.tp, self.fp, self.fn]:
            var.assign(0.0)


    def get_config(self):
      config = super().get_config()
      return config


