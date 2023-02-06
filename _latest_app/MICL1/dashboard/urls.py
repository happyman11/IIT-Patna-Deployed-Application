from django.urls import path

from . import views

urlpatterns = [
    # path('test/', views.test, name='test'),
    # path('home/', views.index, name='index'),
    # path('SGGraph/', views.SGGraph, name='SGGraph'),
    # path('test12/', views.test12, name='test12'),

    #  path('test_AG/', views.test_AG, name='test_AG'),
    #  path('test12_AG/', views.test12_AG, name='test12_AG'),
    # path('AGGraph/', views.AGGraph, name='AGGraph'),

    path('train/', views.train_model, name='train'),
    path('train_model_form/', views.train_model_form, name='train_model_form'),
 
    
]