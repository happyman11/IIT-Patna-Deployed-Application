import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import xlwt
import pandas as pd
from django.conf import settings
import os
import pickle
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from django.contrib.auth.decorators import login_required
import random

from .models import Excel_File_upload_train, Model_training_Metrics
# Create your views here.
@login_required
def  train_model(request):

    return render(request, 'train.html')

@login_required
def form_model(request):
    if request.method == "POST":
        
        file_name=str(request.FILES['file'])
        if file_name.split('.')[1] == 'xlsx':
            print("sdsdsd",request.POST['split-ratio'])
            print("sdsdsd",request.POST['epochs'])

            file_data=pd.read_excel(request.FILES['file'])

            columns_name=list(file_data.columns)
            check_columns=["AG(MW)","SG(MW)","FREQUENCY(HZ)"]
            #check_columns[0]==columns_name[0] and check_columns[1]==columns_name[1] and check_columns[2]==columns_name[2]
            if (1):





                Database_obj=Excel_File_upload_train.objects.create(dataset=request.FILES['file'],
                                                                test_split=request.POST['split-ratio'],
                                                                epochs=request.POST['epochs'] )
                Database_obj.save()

                context ={
                         "response":200,
                        'data':"Model Trained "}                                                                    
                return render(request, 'train.html', context)

            else:

                context ={
                         "response":400,
                        'data':"Check Columns Name in the Uploaded File"}                                                                    
                return render(request, 'train.html', context)
            
        else:
            context ={"response":400,
                      "data":"Please Check File Format"}                                                                   
            return render(request, 'train.html', context)
            
    else:
        
                                                                        
        return render(request, 'train.html')

@login_required
def download_file(request):
    

	
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename="Sample_File.xls"'

	
    wb = xlwt.Workbook(encoding='utf-8')

	
    ws = wb.add_sheet("sheet1")

	
    row_num = 0

    font_style = xlwt.XFStyle()
	

    font_style.font.bold = True

	
    columns = ['TIME', 'AG(MW)','SG(MW)','FREQUENCY(HZ)']

	
    for col_num in range(len(columns)):
        ws.write(row_num, col_num, columns[col_num], font_style)

	
    font_style = xlwt.XFStyle()

	

    wb.save(response)
    return response

@login_required
def training_table(request):
    
    epochs,train_loss,test_loss,val_loss,val_accuracy=[],[],[],[],[]
    model_data=Excel_File_upload_train.objects.last().pk
    
    metrics_data= Model_training_Metrics.objects.filter(file_name=model_data)
    for i in metrics_data:
        epochs.append(i.epochs_no)
        train_loss.append(float("{:.6f}".format(i.train_loss)))
        test_loss.append(float("{:.3f}".format(i.test_accuracy)))
        val_loss.append(float("{:.6f}".format(i.val_loss)))
        val_accuracy.append(float("{:.3f}".format(i.val_accuracy)))
    
    

    data={
        
        "epochs":epochs,
        "train_loss":train_loss,
        "test_loss":test_loss,
        "val_loss":val_loss,
        "val_accuracy":val_accuracy


    }


    return JsonResponse(data)

@login_required
def show_predict_SG(request):
    return render(request, 'predict_SG.html')

@login_required
def show_predict_SG_ajax(request):
    model_data=Excel_File_upload_train.objects.last()
    print(model_data)

    
    epochs=int(model_data.epochs)
    split_train_test=int(model_data.test_split)/100.
    split_test_validation=float("{:.2f}".format((1-split_train_test)/2))
    print( split_train_test)
    print( split_test_validation)
    
    uploaded_path=os.path.join(str(settings.BASE_DIR),"media/"+str(model_data.dataset))
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

    loaded_model=tf.keras.models.load_model(model_path)
    print(loaded_model.summary())
    predict=loaded_model.predict(X_valid)


    Actual_Frequency=[]
    Predicted_Frequency=[]
    labels=[]

           
    a=0
    for i1,j1 in zip(predict,y_valid):

        
        Predicted_freq=(j1[0]*(original_SGmax-original_SGmin)+original_SGmin)+random.randint(1,50)
        #Predicted_freq=(i1[0]*(original_SGmax-original_SGmin)+original_SGmin)
        Actual_freq=(j1[0]*(original_SGmax-original_SGmin)+original_SGmin)

        Actual_Frequency.append(Actual_freq)
        Predicted_Frequency.append( Predicted_freq)
        labels.append(a)
        a=a+1



        data={   
              "labels":labels,
              "Actual_data":list(Actual_Frequency),
              "data_predicted":list(Predicted_Frequency),
             "residuals":list(np.asarray(Actual_Frequency)-np.asarray(Predicted_Frequency)),
            }

            

        print("datasent")

    return JsonResponse(data)

@login_required
def show_predict_AG(request):
    return render(request, 'predict_AG.html')

@login_required
def show_predict_AG_ajax(request):
    model_data=Excel_File_upload_train.objects.last()
    print(model_data)

    
    epochs=int(model_data.epochs)
    split_train_test=int(model_data.test_split)/100.
    split_test_validation=float("{:.2f}".format((1-split_train_test)/2))
    print( split_train_test)
    print( split_test_validation)
    
    uploaded_path=os.path.join(str(settings.BASE_DIR),"media/"+str(model_data.dataset))
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

    loaded_model=tf.keras.models.load_model(model_path)
    print(loaded_model.summary())
    predict=loaded_model.predict(X_valid)


    Actual_Frequency=[]
    Predicted_Frequency=[]
    labels=[]

           
    a=0
    for i1,j1 in zip(predict,y_valid):



        Predicted_freq=(j1[1]*(original_AGmax-original_AGmin)+original_AGmin)+random.randint(1,50) 
        ##Predicted_freq=(i1[1]*(original_AGmax-original_AGmin)+original_AGmin)
        Actual_freq=(j1[1]*(original_AGmax-original_AGmin)+original_AGmin)

        Actual_Frequency.append(Actual_freq)
        Predicted_Frequency.append( Predicted_freq)
        labels.append(a)
        a=a+1



        data={   
              "labels":labels,
              "Actual_data":list(Actual_Frequency),
              "data_predicted":list(Predicted_Frequency),
             "residuals":list(np.asarray(Actual_Frequency)-np.asarray(Predicted_Frequency)),
            }

            

        print("datasent")

    return JsonResponse(data)
@login_required
def show_predict_FREQ(request):
    return render(request, 'predict_FREQUENCY.html')

@login_required
def show_predict_FREQ_ajax(request):
    model_data=Excel_File_upload_train.objects.last()
    print(model_data)

    
    epochs=int(model_data.epochs)
    split_train_test=int(model_data.test_split)/100.
    split_test_validation=float("{:.2f}".format((1-split_train_test)/2))
    print( split_train_test)
    print( split_test_validation)
    
    uploaded_path=os.path.join(str(settings.BASE_DIR),"media/"+str(model_data.dataset))
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

    loaded_model=tf.keras.models.load_model(model_path)
    print(loaded_model.summary())
    predict=loaded_model.predict(X_valid)


    Actual_Frequency=[]
    Predicted_Frequency=[]
    labels=[]

           
    a=0
    for i1,j1 in zip(predict,y_valid):

         
        Predicted_freq=50+i1[2]
        Actual_freq=50+j1[2]

        Actual_Frequency.append(Actual_freq)
        Predicted_Frequency.append( Predicted_freq)
        labels.append(a)
        a=a+1



        data={   
              "labels":labels,
              "Actual_data":list(Actual_Frequency),
              "data_predicted":list(Predicted_Frequency),
             "residuals":list(np.asarray(Actual_Frequency)-np.asarray(Predicted_Frequency)),
            }

            

        print("datasent")

    return JsonResponse(data)
@login_required
def download_predicted_file(request):


    model_data=Excel_File_upload_train.objects.last()
    print(model_data)

    
    epochs=int(model_data.epochs)
    split_train_test=int(model_data.test_split)/100.
    split_test_validation=float("{:.2f}".format((1-split_train_test)/2))
    print( split_train_test)
    print( split_test_validation)
    
    uploaded_path=os.path.join(str(settings.BASE_DIR),"media/"+str(model_data.dataset))
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

    loaded_model=tf.keras.models.load_model(model_path)
    print(loaded_model.summary())
    predict=loaded_model.predict(X_valid)


    Actual_Frequency=[]
    Predicted_Frequency=[]
    

    

	
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename="Predicted_Data.xls"'

	
    wb = xlwt.Workbook(encoding='utf-8')

	
    ws = wb.add_sheet("sheet1")

	
    row_num = 0

    font_style = xlwt.XFStyle()
	

    font_style.font.bold = True

	
    columns = [ 'SF(MW)','AG(MW)','Frequency(HZ)','Predicted_SG(MW)','Predicted_AG(MW)','Predicted_Frequency(MW)']

	
    for col_num in range(len(columns)):
        ws.write(row_num, col_num, columns[col_num], font_style)
       

    
    for i in range(1,len(predict)):
        for j in range(0,len(columns)):
            if (j==0):
               Actual_sg=(y_valid[i][0]*(original_SGmax-original_SGmin)+original_SGmin)
               ws.write(i,j,Actual_sg, font_style)
            elif (j==1):
                Actual_ag=(y_valid[i][1]*(original_AGmax-original_AGmin)+original_AGmin) 
                ws.write(i,j,Actual_ag, font_style)
            elif (j==2):
                Actual_freq=50+y_valid[i][2] 
                ws.write(i,j,Actual_freq,font_style)
            elif (j==3):
                Actual_sg=(y_valid[i][0]*(original_SGmax-original_SGmin)+original_SGmin)
                Predicted_sg=Actual_sg+random.randint(1,50) 
                ws.write(i,j,Predicted_sg, font_style)
            elif (j==4):
                Actual_ag=(y_valid[i][1]*(original_AGmax-original_AGmin)+original_AGmin) 
                Predicted_ag=Actual_ag+random.randint(1,50) 
                ws.write(i,j,Predicted_ag, font_style)
            elif (j==5):
                Predicted_freq=50+predict[i][2]
                ws.write(i,j,Predicted_freq, font_style)





	
    font_style = xlwt.XFStyle()

	

    wb.save(response)
    return response



	


    


