from sklearn.utils.class_weight import compute_class_weight
import numpy as np 
import tensorflow as tf 


def calculate_class_weights(dataset):
    unbatched = dataset.unbatch()
    y_train = []
    for _, y in unbatched:
        y_train.append(y.numpy())
    y_train = np.array(y_train)
    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    # Convert class weights to dictionary
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    return class_weights_dict, len(y_train)



def get_lr_shedular(cfg, len_y_train, type, mode):
     ##If not opther model introduce the lr shedular 
    steps_per_epoch = len_y_train // cfg.batch_size
    if mode == 'train':
        decay_steps = steps_per_epoch * cfg.epochs
    else: 
        decay_steps = steps_per_epoch * cfg.finetune.epochs

    if type == 'cosine':
        return tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=cfg.lr if cfg.mode=='train' else cfg.finetune.lr,  # Start low to avoid instability
                                                         decay_steps=decay_steps,  # Number of steps before the first decay
                                                         alpha=cfg.min_lr  # Minimum learning rate to prevent model collapse
                                                         )
    elif type=='poly':
        return tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=cfg.lr if cfg.mode=='train' else cfg.finetune.lr,
                                                             decay_steps= decay_steps,
                                                             end_learning_rate=cfg.min_lr,
                                                             power=2.0)
    