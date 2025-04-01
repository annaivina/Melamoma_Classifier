import tensorflow as tf
import os
import datetime
import datetime 
import logging 

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def get_callbacks(config, experiment_name='', model_name='', use_lrshedular=False):
    
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
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=config['callbacks']['monitor'], patience=3)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir+'best_model.keras', monitor=config['callbacks']['monitor'])
    csv_logger = tf.keras.callbacks.CSVLogger(filename=log_dir+'training.csv', separator='.')


    callbacks = [early_stopping, model_checkpoint, csv_logger]

    if use_lrshedular == True:
        lr_shedular = tf.keras.callbacks.ReduceLROnPlateau(monitor=config['callbacks']['monitor'], factor=0.5, patience=3, min_lr=config['callbacks']['min_lr'])
        callbacks.append(lr_shedular)


    return callbacks

    




    


