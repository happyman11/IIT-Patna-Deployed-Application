from django.db import models

# Create your models here.

class Excel_File_upload_train(models.Model):

    
    dataset = models.FileField(upload_to='media/')
    trained = models.BooleanField(default=False)
    epochs=models.IntegerField()
    test_split=models.IntegerField()

    def __str__(self):
        return "%s ||%s || %s || %s" % (self.dataset, self.trained,self.epochs,self.test_split)

class Model_training_Metrics(models.Model):
    file_name = models.ForeignKey(Excel_File_upload_train, on_delete=models.CASCADE)
    epochs_no=models.IntegerField(default=0)
    train_loss=models.FloatField(default=0)
    test_accuracy=models.FloatField(default=0)

    test_loss=models.FloatField(default=0)
    test_accuracy=models.FloatField(default=0)

    val_loss=models.FloatField(default=0)
    val_accuracy=models.FloatField(default=0)

    def __str__(self):
        return "%s ||%s || %s || %s" % (self.file_name, self.epochs_no,self.train_loss,self.test_loss)


