from gdstorage.storage import GoogleDriveStorage
from django.db import models
# Define Google Drive Storage
gd_storage = GoogleDriveStorage()
class Cloud(models.Model):
    upload = models.FileField(upload_to='./', storage=gd_storage)