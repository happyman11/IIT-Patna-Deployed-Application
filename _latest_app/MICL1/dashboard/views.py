from django.shortcuts import render
from .models import Document,Documentpred
from django.http import JsonResponse
from django.conf import settings
import pandas as pd
import os
import pickle
import time
import numpy as np
# import tensorflow as tf
import math

# def  index(request):
    
#     return render(request, 'index.html')
# # Create your views here.

# def  test(request):

#     if request.method=='POST':
#         file_name=str(request.FILES['file'])
#         if file_name.split('.')[1] == 'xlsx':

#             Document_obj=Document.objects.create(docfile=request.FILES['file'])
#             Document_obj.save()

#         else:
            
#             print("file Type")

    
#         return render(request, 'test.html')
    
#     else:
#         # print("not Post METhor")
    
#         return render(request, 'test.html')




# def  test12(request):

#     if request.method=='POST':
#         file_name=str(request.FILES['file'])
#         if file_name.split('.')[1] == 'xlsx':

#             Document_obj=Documentpred.objects.create(docfilepred=request.FILES['file'])
#             Document_obj.save()
#             print(file_name)

#         else:
            
#             print("file Type")

    
#         return render(request, 'test.html')
    
#     else:
#         # print("not Post METhor")
    
#         return render(request, 'test.html')

# def SGGraph(requesf):
#     print("ok")



#     filesname=Documentpred.objects.filter(done =False).first()
#     print(filesname.docfilepred)



#     scaler_path=os.path.join(str(settings.BASE_DIR),'dashboard/Scaler_pickle/MinMaxScaler.pickle')
#     uploaded_path=os.path.join(str(settings.BASE_DIR),"media/"+str(filesname.docfilepred))

#     model_path=os.path.join(str(settings.BASE_DIR),"dashboard/saved_model-20230103T111232Z-001/saved_model")
#     # file_path=os.path.join(settings.MEDIA_ROOT,str(instance.docfile))
        
       
#     with open(scaler_path, 'rb') as f:
#         scaler = pickle.load(f)
            
       
#         data_train=pd.read_excel(uploaded_path)
#         print(data_train.columns)

#         data_train["Date"]=pd.to_datetime(data_train["Date"])
#         data_train["Unit_Running"]=data_train["Unit_Running"].astype(int)
#         data_train["Avg. Frequency"]=data_train["Avg. Frequency"].astype(float)
        



#         data_train=data_train.dropna(axis=0)

#         data_train['DayofWeek']=data_train['Date'].dt.dayofweek
#         data_train['Month_number']=data_train['Date'].dt.month

#         week_day_number_name={
#                  0: "Monday",
#                  1: "Tuesday",
#                  2: "Wednesday",
#                  3: "Thrusday",
#                  4: "Friday",
#                  5: "Saturday",
#                  6: "Sunday"
#                 }

#         day_name=[ week_day_number_name[i] for i in data_train.DayofWeek]

#         data_train["Day Name"]=day_name

#             # print(data_train.head())

#         data_selected=data_train[['SG(MW)','AG(MW)','Avg. Frequency','DayofWeek','Month_number','Unit_Running']].copy()

       
#         scaled_model=scaler.fit(data_selected)
#         scaled_data=scaled_model.transform(data_selected)

#         X= []
#         y = []
#         timestamp=30
#         length=scaled_data.shape[0]
            
#         for i in range(timestamp, length):
#             X.append(scaled_data[i-30:i])
#             y.append(scaled_data[i])
#         X, y = np.array(X), np.array(y)

#         X = np.reshape(X, (X.shape[0], X.shape[1], 6))

#         print("Dataset with lag of 30 (x): ")
#         print(X[0])
#         print("Target value")
#         print(y[0])

#         time=list(data_train['Time'].values)
#         SG=list(data_train['SG(MW)'].values)
#         AG=list(data_train['AG(MW)'].values)
#         Avg_frequency=[x for x in list(data_train['Avg. Frequency'].values)]

#         loaded_model=tf.keras.models.load_model(model_path)
#         prediction=loaded_model.predict(X)
#         predicted_SG=[]
#         predicted_AG=[]
#         predicted_freq=[]
#         predicted_freq1=[]

#         for i in Avg_frequency:

#             if i > 50:
#                 predicted_freq.append(i+0.006)
#                 print(i,i+0.006)

#             else:
#                 predicted_freq.append(i-0.006)
#                 print(i,i-0.006)

#         for i in prediction:
#             upscale=scaler.inverse_transform(i.reshape(1,-1))
#             predicted_SG.append(float("{:.3f}".format(upscale[0][0])))
#             predicted_AG.append(float("{:.3f}".format(upscale[0][1])))
#             predicted_freq1.append(float("{:.3f}".format(upscale[0][2])))
#             #print(float("{:.3f}".format(upscale[0][2])))
            
#             #print(int(upscale[0][0]))


#     data={
#             "labels": time,
#             "data":SG,
#             "data_predicted":list(predicted_SG),
#             "data1":AG,
#             "data_predicted1":list(predicted_AG),
#             "data2":Avg_frequency,
#             "data_predicted2":list(predicted_freq),
           
           
           
            
#             }
#     print("data sending")
#     return JsonResponse(data)



# def  test_AG(request):

#     if request.method=='POST':
#         file_name=str(request.FILES['file'])
#         if file_name.split('.')[1] == 'xlsx':

#             Document_obj=Document.objects.create(docfile=request.FILES['file'])
#             Document_obj.save()

#         else:
            
#             print("file Type")

    
#         return render(request, 'test_AG.html')
    
#     else:
#         # print("not Post METhor")
    
#         return render(request, 'test_AG.html')

# def  test12_AG(request):

#     if request.method=='POST':
#         file_name=str(request.FILES['file'])
#         if file_name.split('.')[1] == 'xlsx':

#             Document_obj=Documentpred.objects.create(docfilepred=request.FILES['file'])
#             Document_obj.save()
#             print(file_name)

#         else:
            
#             print("file Type")

    
#         return render(request, 'test_AG.html')
    
#     else:
#         # print("not Post METhor")
    
#         return render(request, 'test_AG.html')


# def AGGraph(requesf):
#     print("ok")



#     filesname=Documentpred.objects.get(done =False)
#     print(filesname.docfilepred)



#     scaler_path=os.path.join(str(settings.BASE_DIR),'dashboard/Scaler_pickle/MinMaxScaler.pickle')
#     uploaded_path=os.path.join(str(settings.BASE_DIR),"media/"+str(filesname.docfilepred))

#     model_path=os.path.join(str(settings.BASE_DIR),"dashboard/saved_model-20230103T111232Z-001/saved_model")
#     # file_path=os.path.join(settings.MEDIA_ROOT,str(instance.docfile))
        
       
#     with open(scaler_path, 'rb') as f:
#         scaler = pickle.load(f)
            
       
#         data_train=pd.read_excel(uploaded_path)
#         print(data_train.columns)

#         data_train["Date"]=pd.to_datetime(data_train["Date"])
#         data_train["Unit_Running"]=data_train["Unit_Running"].astype(int)
        



#         data_train=data_train.dropna(axis=0)

#         data_train['DayofWeek']=data_train['Date'].dt.dayofweek
#         data_train['Month_number']=data_train['Date'].dt.month

#         week_day_number_name={
#                  0: "Monday",
#                  1: "Tuesday",
#                  2: "Wednesday",
#                  3: "Thrusday",
#                  4: "Friday",
#                  5: "Saturday",
#                  6: "Sunday"
#                 }

#         day_name=[ week_day_number_name[i] for i in data_train.DayofWeek]

#         data_train["Day Name"]=day_name

#             # print(data_train.head())

#         data_selected=data_train[['SG(MW)','AG(MW)','DayofWeek','Month_number','Unit_Running']].copy()

       
#         scaled_model=scaler.fit(data_selected)
#         scaled_data=scaled_model.transform(data_selected)

#         X= []
#         y = []
#         timestamp=30
#         length=scaled_data.shape[0]
            
#         for i in range(timestamp, length):
#             X.append(scaled_data[i-30:i])
#             y.append(scaled_data[i])
#         X, y = np.array(X), np.array(y)

#         X = np.reshape(X, (X.shape[0], X.shape[1], 5))

#         print("Dataset with lag of 30 (x): ")
#         print(X[0])
#         print("Target value")
#         print(y[0])

#         time=list(data_train['Time'].values)
#         SG=list(data_train['AG(MW)'].values)

#         loaded_model=tf.keras.models.load_model(model_path)
#         prediction=loaded_model.predict(X)
#         predicted_SG=[]
#         for i in prediction:
#             upscale=scaler.inverse_transform(i.reshape(1,-1))
#             predicted_SG.append(int(upscale[0][1]))
#             #print(int(upscale[0][0]))


#     data={
#             "labels": time,
#             "data":SG,
#             "data_predicted":list(predicted_SG)
           
           
            
#             }
#     print("data sending")
#     return JsonResponse(data)

def  train_model(request):

    if request.method == "POST":
        print("POST FORM")
        file_name=str(request.FILES['file'])
        if file_name.split('.')[1] == 'xlsx':
            print("sdsdsd",request.data.POST('split-ratio'))
            print("sdsdsd",request.data.POST('epochs'))
            return render(request, 'train.html')
        else:
            print("Incorrect FIle data")
            return render(request, 'train.html')
    else:
        print("POST FORM")
        return render(request, 'train.html')

    
    return render(request, 'train.html')

def train_model_form(request):
    if request.method == "POST":
        print("POST FORM")
        file_name=str(request.FILES['file'])
        if file_name.split('.')[1] == 'xlsx':
            print("sdsdsd",request.data.POST('split-ratio'))
            print("sdsdsd",request.data.POST('epochs'))
            return render(request, 'train.html')
        else:
            print("Incorrect FIle data")
            return render(request, 'train.html')
    else:
        print("POST FORM")
        return render(request, 'train.html')

# Create your views here.
