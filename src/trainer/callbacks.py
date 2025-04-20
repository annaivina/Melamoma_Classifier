import tensorflow as tf
import os
import datetime
import datetime 
import logging 

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class SaveEachEpoch(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, monitor='val_accuracyB'):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs.get(self.monitor)
        if metric_value is not None:
            filename = f"model_epoch{epoch+1:02d}_{self.monitor}_{metric_value:.4f}.weights.h5"
            filepath = os.path.join(self.save_dir, filename)
            self.model.save_weights(filepath)
            print(f"🔸 Saved weights to: {filepath}")



def get_callbacks(cfg, experiment_name='', model_name='', use_lrshedular=False):
    
    if experiment_name=='' and model_name=='':
        logging.error("You didnt specify experiment name or model. Plase make sure you specify both")
        return None
    
    try:
        os.makedirs(experiment_name,exist_ok=True)
        log_dir = os.path.join(experiment_name,"logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        checkpoint_dir = os.path.join(experiment_name,"checkpoints",model_name)
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    except OSError as e:
        logging.error("Failed to create directories: {e}")
        return None
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=cfg.monitor, patience=cfg.patience, mode=cfg.mode_save, restore_best_weights=cfg.restore_best_weights)#restore best weights shall be False if use the save_each_epoch
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir+'best_model.keras', monitor=cfg.monitor)
    save_each_epoch = SaveEachEpoch(save_dir=checkpoint_dir, monitor=cfg.monitor)
    csv_logger = tf.keras.callbacks.CSVLogger(filename=log_dir+'training.csv', separator='.')


    callbacks = [early_stopping, save_each_epoch, csv_logger]

    if use_lrshedular == True:
        lr_shedular = tf.keras.callbacks.ReduceLROnPlateau(monitor=cfg.monitor, factor=0.5, patience=cfg.patience, min_lr=cfg.min_lr)
        callbacks.append(lr_shedular)


    return callbacks

    




    


