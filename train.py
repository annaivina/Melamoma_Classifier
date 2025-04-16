import comet_ml
import tensorflow as tf 
import numpy as np
from src.trainer.callbacks import get_callbacks
from src.models.custom_cnn import MelanomaClassifier
from src.models.ViT import VIT
from src.utils import  calculate_class_weights, get_lr_shedular, make_model 
from src.models.eff_net import EfficientNetBase
from src.trainer.model_fit import Classifier 
from src.metrics.custom_metrics import BalancedAccuracy, CustomF1Score 
from src.load_config import load_config 
from src.data_loader import DataLoader 
import os
import logging 
import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", 
                    handlers=[logging.StreamHandler()])


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def pipeline(cfg: DictConfig) -> None:
    
    logging.info(f"Running {cfg.mode} experiment: {cfg.experiment_name}")
    print(OmegaConf.to_yaml(cfg))
  
    
    #Load the datasets
    logging.info("Loading the data from the DataLoader")
    data_loader = DataLoader(dataset=cfg.dir_path,
                             csv_train=cfg.csv_train,
                             csv_test=cfg.csv_test,
                             batch_size=cfg.batch_size,
                             img_size=cfg.img_size,
                             train_val_split=cfg.train_val_split,
                             eff_net = False if cfg.model.name=='customCNN' else True)
    
    train_data, valid_data, _ = data_loader.load_data() #We dont need now the test data 

    experiment = comet_ml.Experiment(api_key = os.getenv("COMET_API_KEY"),
                                              project_name=cfg.experiment_name,
                                              auto_param_logging=True,
                                              auto_metric_logging=True)
    

    #Load the models of interes: CustomCNN, EffB2 - Transfer Learning or VIT
    if cfg.model.name == 'customCNN':
        model = MelanomaClassifier(cfg)
    
    elif cfg.model.name == 'effnetb2':
        model = EfficientNetBase(cfg)
    elif cfg.model.name == 'transformer':
        model = make_model(VIT(cfg), MelanomaClassifier(cfg))
       

    ##We can check the summary of the model 
    _ = model(tf.random.normal((1, 256, 256, 3)))
    model.summary()
        

    #Extract class weights and len of y_train for weightitngn
    class_weights_dict, len_y_train = calculate_class_weights(train_data)

    #Define metrics to use:
    metrics = [BalancedAccuracy(name='accuracyB'), tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(curve="ROC")]#While using custom loop you need to specify fully your accuracy 
   

    ##mode=train (stage 1 for transfer learning)
    if cfg.mode == 'train':
        logging.info("The mode is train. Start the training cycle")
        if cfg.model.name == 'effnetb2':
            model.feature_extractor.trainable = False #Just to make sure its not going to train
         

        
        model_trainer = Classifier(model, experiment_name=cfg.experiment_name)

        lr_schedule = get_lr_shedular(cfg, len_y_train, type=cfg.lr_shedule, mode=cfg.mode)
       
        callbacks = get_callbacks(cfg, experiment_name=cfg.experiment_name, model_name=cfg.model.name, use_lrshedular=False)
            

        logging.info("Compiling the model")
        model_trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule if cfg.model.name=='customCNN' or cfg.model.name=='transformer' else cfg.lr),
                              loss=tf.keras.losses.BinaryFocalCrossentropy(label_smoothing=cfg.label_smoothing, gamma=cfg.gamma),
                              metrics_list=metrics)
        
        logging.info("Start fitting")
        with experiment.train():
            model_trainer.fit(train_data, 
                              validation_data=valid_data,
                              epochs=cfg.epochs, 
                              class_weight = class_weights_dict,
                              callbacks=callbacks)
        
    elif cfg.mode == 'fine_tune':
        #define the checkpoint dir
        logging.info("The mode is fine-tune, start the cycle")
        checkpoint_dir = os.path.join(cfg.experiment_name,"checkpoints",cfg.model.name,"best_model.keras")
        model_from_checkpoint = tf.keras.models.load_model(checkpoint_dir,custom_objects={'BalancedAccuracy': BalancedAccuracy})

        logging.info(f"Unfreezing {cfg.finetune.num_layers} layers of the model")
        model_from_checkpoint.unfreeze_top_layers(num_layers=cfg.finetune.num_layers)
        model_trainer = Classifier(model_from_checkpoint, experiment_name=cfg.experiment_name)

        lr_schedule = get_lr_shedular(cfg, len_y_train, type=cfg.finetune.lr_shedule, mode=cfg.mode)

        logging.info("Compiling the model to fine-tune")
        model_trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule),
                              loss=tf.keras.losses.BinaryFocalCrossentropy(label_smoothing=cfg.finetune.label_smoothing, gamma=cfg.finetune.gamma),
                              metrics_list=metrics)
        
        
        callbacks = get_callbacks(experiment_name=cfg.experiment_name, model_name=cfg.model.name, use_lrshedular=False)
        logging.info("Start the fine-tuning...")
        model_trainer.fit(train_data, 
                          validation_data=valid_data,
                          epochs=cfg.finetune.epochs, 
                          class_weight = class_weights_dict,
                          callbacks=callbacks)
        


if __name__ == "__main__":
    pipeline()

    

        
    

    

    

    



