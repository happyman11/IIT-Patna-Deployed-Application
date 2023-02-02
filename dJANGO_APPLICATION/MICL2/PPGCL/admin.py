from django.contrib import admin

# Register your models here.

from .models import Excel_File_upload_train,Model_training_Metrics



admin.site.register(Excel_File_upload_train)
admin.site.register(Model_training_Metrics)