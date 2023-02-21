from django.db import models

# Create your models here.
#class FileUpload(models.Model):
#    file = models.FileField(null=True,upload_to="",blank=True)


class App(models.Model):
    def __str__(self):
        return str(self)

class Info(models.Model):
    id = models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')
    image = models.ImageField(upload_to=None) # 이미지
    result = models.CharField(max_length=30) # 모델 돌린 결과
