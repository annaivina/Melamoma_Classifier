import tensorflow as tf 
import numpy as np
from src.trainer.callbacks import get_callbacks
from src.models.custom_cnn import MelanomaClassifier
from src.utils import  calculate_class_weights, get_lr_shedular
#from src.models.eff_net import EffNetB0
from src.trainer.model_fit import Classifier 
from src.metrics.custom_metrics import BalancedAccuracy, CustomF1Score 
from src.load_config import load_config 
from src.data_loader import DataLoader 
import argparse 
from tensorflow.keras.utils import plot_model






def main(mode, experiment_name, config_path):
    
    config = load_config(config_path)
    
    #Load the datasets
    data_loader = DataLoader(dataset=config['dataset']['dir_path'],
                             csv_train=config['dataset']['csv_train'],
                             csv_test=config['dataset']['csv_test'],
                             batch_size=config['dataset']['batch_size'],
                             img_size=config['dataset']['img_size'],
                             train_val_split=config['dataset']['train_val_split'],
                             eff_net = False if config['dataset']['model_name']=='customCNN' else True)
    
    train_data, valid_data, _ = data_loader.load_data() #We dont need now the test data 

    #Load the models of interes: CustomCNN, EffB2 - Transfer Learning or VIT
    if config['dataset']['model_name'] == 'customCNN':
        model = MelanomaClassifier(config)
    
    elif config['dataset']['model_name'] == 'effnetB2':
        #model = EffNetB0(config)
        print('not yet done')


    model_trainer = Classifier(model, experiment_name=experiment_name)

    #Extract class weights and len of y_train for weightitngn
    class_weights_dict, len_y_train = calculate_class_weights(train_data)

    #Define the Callbacks:
    if mode == 'train':
        if config['dataset']['model_name'] == 'customCNN':
            callbacks = get_callbacks(config, experiment_name=experiment_name, model_name=config['dataset']['model_name'], use_lrshedular=False)
            
        else:
            #Need to check that cosine is worse then usual lr shedular 
            callbacks = get_callbacks(experiment_name=experiment_name, model_name=config['dataset']['model_name'], use_lrshedular=True)
            
        
        #Define metrics to use:
        metrics = [BalancedAccuracy(name='accuracyB'), tf.keras.metrics.BinaryAccuracy()]#While using custom loop you need to specify fully your accuracy 
        #Lr schedular
        lr_schedule = get_lr_shedular(config, len_y_train, type=config['params']['lr_shedule'])
        
        model_trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                              loss=tf.keras.losses.BinaryFocalCrossentropy(),
                              metrics_list=metrics)
        
        
        model_trainer.fit(train_data, 
                          validation_data=valid_data,
                          epochs=config['params']['epochs'], 
                          class_weight = class_weights_dict,
                          callbacks=callbacks)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate different models for Melanoma classification")
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--experiment_name", type=str, default="test")
    parser.add_argument("--config_path", type=str, default='configs/cnn_clussifier.yaml')

    args = parser.parse_args()
    
    main(mode=args.mode, experiment_name=args.experiment_name, config_path=args.config_path)


    


    

        
    

    

    

    



