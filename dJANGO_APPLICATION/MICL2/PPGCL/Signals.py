from django.db.models.signals import post_save
from .models import Excel_File_upload_train
from django.dispatch import receiver
from django.conf import settings
import pandas as pd
import os
import pickle
import time
import numpy as np
import tensorflow as tf
from .utilities import *
from sklearn.model_selection import train_test_split

@receiver(post_save, sender=Excel_File_upload_train)
def create_profile(sender, instance, created, **kwargs):
    if created:
        print(instance.pk)

        
        uploaded_path=os.path.join(str(settings.BASE_DIR),"media/"+str(instance.dataset))
        epochs=int(instance.epochs)
        split_train_test=int(instance.test_split)/100.
        split_test_validation=float("{:.2f}".format((1-split_train_test)/2))
        print( split_train_test)
        print( split_test_validation)
       
        model_path=os.path.join(str(settings.BASE_DIR),"PPGCL/saved_model_frequency_mannual_scaling.h5")
    
       
        data_train=pd.read_excel(uploaded_path)

        data_train=data_train.dropna(axis=0)

            

        data_selected=data_train[['SG(MW)','AG(MW)','Avg. Frequency']].copy()

       
        data_selected['Avg. Frequency']=data_selected['Avg. Frequency'].astype(float)

        original_SGmin,original_SGmax=335.78,1817.95
        original_AGmin,original_AGmax=339.74,1835.41

        data_selected['SG(MW)']=(data_selected['SG(MW)']-original_SGmin)/(original_SGmax-original_SGmin)
        data_selected['AG(MW)']=(data_selected['AG(MW)']-original_AGmin)/(original_AGmax-original_AGmin)
        data_selected['Avg. Frequency']=50-data_selected['Avg. Frequency']


        X= []
        y = []
        timestamp=32
        scaled_data=np.asarray(data_selected.values.tolist())
        length=scaled_data.shape[0]
            
        for i in range(timestamp, length):
            X.append(scaled_data[i-timestamp:i])
            y.append(scaled_data[i])
            
        X, y = np.array(X), np.array(y)

        X = np.reshape(X, (X.shape[0], X.shape[1], 3))

        X_train,X_data,y_train,y_data=train_test_split(X,y,train_size=split_train_test,random_state=42)
            
        X_test,X_valid,y_test,y_valid=train_test_split(X_data,y_data,test_size=split_test_validation,random_state=42)

            

        loaded_model=tf.keras.models.load_model(model_path)
        print(loaded_model.summary())
        callbacks= My_callback(int(instance.pk))

        hist=loaded_model.fit(X_train, y_train, batch_size=32, epochs=epochs,callbacks=[callbacks],validation_data=(X_valid, y_valid))

        loaded_model.save(model_path)



       
        