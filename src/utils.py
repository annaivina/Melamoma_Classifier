from sklearn.utils.class_weight import compute_class_weight
from lr_schedule.warmup_cosine import WarmupCosineDecay
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
        decay_steps = (steps_per_epoch * cfg.epochs) + 1 #what ever reason the is around problem 
    elif mode=='finetune': 
        decay_steps = steps_per_epoch * cfg.finetune.epochs +1 

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
    elif type=='warmupcosine':
        return WarmupCosineDecay(total_steps=decay_steps, 
                                 warmup_steps=steps_per_epoch*3, 
                                 hold_steps=steps_per_epoch*2, 
                                 start_lr=1e-6, 
                                 target_lr=5e-6, 
                                 alpha=0.1)
    
def make_model(model_1 = tf.keras.models.Model, model_2=tf.keras.models.Model):
    #Take feature extractor form the CNN
    model_2.compile(optimizer='adam', loss=tf.keras.losses.BinaryFocalCrossentropy(label_smoothing=0.0, gamma=3),metrics=['accuracy'])
    _ = model_2(tf.zeros((1,256,256,3)), training=False)
    model_2.load_weights('images/best_model_cnn_custom_epoch8.weights.h5')
    feature_extr = model_2.FeatureExtractor
    feature_extr.trainable = False

    #TRake ViT and combine 
    input = tf.keras.layers.Input(shape=(256,256,3))
    x = feature_extr(input)
    output = model_1(x)
        
    return tf.keras.models.Model(inputs=input, outputs=output)
