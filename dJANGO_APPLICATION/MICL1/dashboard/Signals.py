from django.db.models.signals import post_save
from .models import Document
from django.dispatch import receiver
from django.conf import settings
import pandas as pd
import os
import pickle
import time
import numpy as np
import tensorflow as tf 




import pandas as pd

@receiver(post_save, sender=Document)
def create_profile(sender, instance, created, **kwargs):
    if created:

       
       
        file_path=os.path.join(settings.MEDIA_ROOT,str(instance.docfile))
        # print("filepath:::",file_path)
        with open("C:/Users/acer/Desktop/dJANGO_APPLICATION/MICL1/dashboard/Scaler_pickle/MinMaxScaler.pickle", 'rb') as f:
            scaler = pickle.load(f)
            
            PATH= "C:/Users/acer/Desktop/dJANGO_APPLICATION/MICL1/media/"+str(instance.docfile)
            data_train=pd.read_excel(PATH)
            # print(data_train.head())

            data_train["Date"]=pd.to_datetime(data_train["Date"])
            data_train["Unit_Running"]=data_train["Unit_Running"].astype(int)

            data_train=data_train.dropna(axis=0)

            data_train['DayofWeek']=data_train['Date'].dt.dayofweek
            data_train['Month_number']=data_train['Date'].dt.month

            week_day_number_name={
                 0: "Monday",
                 1: "Tuesday",
                 2: "Wednesday",
                 3: "Thrusday",
                 4: "Friday",
                 5: "Saturday",
                 6: "Sunday"
                }

            day_name=[ week_day_number_name[i] for i in data_train.DayofWeek]

            data_train["Day Name"]=day_name

            # print(data_train.head())

            data_selected=data_train[['SG(MW)','AG(MW)','DayofWeek','Month_number','Unit_Running']].copy()

       
            scaled_model=scaler.fit(data_selected)
            scaled_data=scaled_model.transform(data_selected)

            X= []
            y = []
            timestamp=30
            length=scaled_data.shape[0]
            
            for i in range(timestamp, length):
                X.append(scaled_data[i-30:i])
                y.append(scaled_data[i])
            X, y = np.array(X), np.array(y)

            X = np.reshape(X, (X.shape[0], X.shape[1], 5))

            # print("Dataset with lag of 30 (x): ")
            # print(X[0])
            # print("Target value")
            # print(y[0])

            loaded_model=tf.keras.models.load_model("C:/Users/acer/Desktop/dJANGO_APPLICATION/MICL1/dashboard/saved_model-20230103T111232Z-001/saved_model")
            print(loaded_model.summary())
            hist=loaded_model.fit(X, y, batch_size=32, epochs=10)

            loaded_model.save("C:/Users/acer/Desktop/dJANGO_APPLICATION/MICL1/dashboard/saved_model-20230103T111232Z-001/saved_model", include_optimizer="True")

