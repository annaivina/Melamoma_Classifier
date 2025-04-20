##Took is from https://arxiv.org/pdf/1608.03983
import tensorflow as tf 

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, warmup_steps, hold_steps, start_lr, target_lr, alpha=0.0, name=None):
        super().__init__()
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupCosineDecay"):
            step = tf.cast(step, tf.float32)

            def warmup_fn():
                
                warmup_steps = tf.cast(self.warmup_steps, tf.float32)
                step_ratio = step / warmup_steps
                
                return self.start_lr + 0.5 * (self.target_lr - self.start_lr) * (1 - tf.cos(np.pi * step_ratio))

            def hold_fn():
                return tf.constant(self.target_lr, dtype=tf.float32)

            def decay_fn():
                decay_steps = self.total_steps - self.warmup_steps - self.hold_steps
                decay_progress = (step - self.warmup_steps - self.hold_steps) / decay_steps
                cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_progress))
                return self.target_lr * ((1 - self.alpha) * cosine_decay + self.alpha)

            return tf.case([
                (step < self.warmup_steps, warmup_fn),
                (step < self.warmup_steps + self.hold_steps, hold_fn)
            ], default=decay_fn)

    def get_config(self):
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "hold_steps": self.hold_steps,
            "start_lr": self.start_lr,
            "target_lr": self.target_lr,
            "alpha": self.alpha,
            "name": self.name,
        }

