from django.db import models

# Create your models here.
from django.db import models

class Document(models.Model):
    docfile = models.FileField(upload_to='media/')

class Documentpred(models.Model):
    docfilepred = models.FileField(upload_to='media/pred')
    created_at = models.DateTimeField(auto_now_add=True)
    done = models.BooleanField(default=False)

    class Meta:
        abstract:True

class Excel_File_upload_train(models.Model):
    dataset = models.FileField(upload_to='media/')
    trained = models.BooleanField(default=False)
    epochs=models.IntegerField()
    test_split=models.IntegerField()

    def __str__(self):
        return "%s ||%s || %s || %s" % (self.dataset, self.trained,self.epochs,self.test_split)

class Model_training_Metrics(models.Model):
    file_name = models.ForeignKey(Excel_File_upload_train, on_delete=models.CASCADE)
    epochs_no=models.IntegerField()
    loss=models.FloatField()
    accuracy=models.FloatField()

    def __str__(self):
        return "%s ||%s || %s || %s" % (self.file_name, self.epochs_no,self.loss,self.test_split)





    