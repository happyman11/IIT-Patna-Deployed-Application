import tensorflow as tf
from .models import Excel_File_upload_train, Model_training_Metrics
class My_callback(tf.keras.callbacks.Callback):
    def __init__(self,id):

        self.id=id
        self.obj=Excel_File_upload_train.objects.get(pk=self.id)
        self.epoch_count=1
        
    def on_epoch_end(self,batch,logs=None):
        loss_mse=logs['loss']
        accuracy=logs['accuracy']

        val_loss_mse=logs['val_loss']
        val_accuracy=logs['val_accuracy']
        
        

        Model_training_Metrics_object=Model_training_Metrics(file_name=self.obj,
                                                             epochs_no=self.epoch_count,
                                                             train_loss=loss_mse,
                                                             test_accuracy=accuracy,
                                                             val_loss=val_loss_mse,
                                                             val_accuracy= val_accuracy)
        Model_training_Metrics_object.save()
        self.epoch_count=self.epoch_count+1

        
