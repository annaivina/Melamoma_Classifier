import comet_ml
import tensorflow as tf
import os 



class Classifier(tf.keras.Model):
    def __init__(self, model, experiment_name=''):
        super(Classifier,self).__init__(name='CNNTrainer')
        self.model = model
        self.experiment_name= experiment_name
        

    
    def compile(self, optimizer, loss, metrics_list):
        super(Classifier,self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self._metrics = metrics_list

    @property
    def metrics(self):
        return self._metrics
    
    def train_step(self, data):
        #Handling the sace where sample weight is calculated because the fit is done using calculate_weights
        #The dataloader automatically will calculate this 
        
        if len(data)== 3:
             X , y, sample_weight = data
        elif len(data) == 2:
             X, y = data
             sample_weight = None

        with tf.GradientTape() as record:
            y_pred = self.model(X, training=True)
            loss = self.loss(y, y_pred, sample_weight = sample_weight)

        training_vars = self.trainable_variables
        gradients = record.gradient(loss, training_vars)

        self.optimizer.apply_gradients(zip(gradients, training_vars))

        for metric in self.metrics:
            metric.update_state(y, y_pred, sample_weight = sample_weight)
           

        result = {'loss': loss,  **{m.name: m.result() for m in self.metrics}}#so we can log it into comet
        
        return result
    
    def test_step(self, data):
        if len(data)== 3:
             X , y, sample_weight = data
        elif len(data) == 2:
             X, y = data
             sample_weight = None

        y_pred = self.model(X, training=False)
        loss = self.loss(y, y_pred, sample_weight = sample_weight)


        for metric in self.metrics:
            metric.update_state(y, y_pred, sample_weight = sample_weight)
         
        
        result = {'loss': loss,  **{m.name: m.result() for m in self.metrics}}
        
        return result


    

