from django.urls import path

from . import views

urlpatterns = [
   

    path('train/', views.train_model, name='train'),
    path('form_model/', views.form_model, name='form_model'),
    path('form_model_ajax/', views.form_model, name='form_model_ajax'),
    path('download_file/', views.download_file, name='download_file'),
    path('training_table/', views.training_table, name='training_table'),
    
    path('show_predict_SG/', views.show_predict_SG, name='show_predict_SG'),
    path('show_predict_AG/', views.show_predict_AG, name='show_predict_AG'),
    path('show_predict_FREQ/', views.show_predict_FREQ, name='show_predict_FREQ'),
    
    path('show_predict_FREQ_ajax/', views.show_predict_FREQ_ajax, name='show_predict_FREQ_ajax'),
    path('show_predict_AG_ajax/', views.show_predict_AG_ajax, name='show_predict_AG_ajax'),
    path('show_predict_SG_ajax/', views.show_predict_SG_ajax, name='show_predict_SG_ajax'),

    path('download_predicted_file/', views. download_predicted_file, name='download_predicted_file'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    
    

   
    
    
    
   
    
    
]