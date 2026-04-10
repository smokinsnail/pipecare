from django.db import models


# Create your models here.
class PicUpload(models.Model):
    imageFile = models.ImageField(upload_to = 'pic_upload', blank=True)

