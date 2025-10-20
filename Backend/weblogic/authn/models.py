from django.db import models
from django.contrib.auth.models import AbstractUser
import hashlib
# Create your models here.

class User(AbstractUser):
    hashed_username = models.CharField(max_length=255, blank=True, null=True)

class customuser(AbstractUser):
    hashed_username = models.CharField(max_length=64, unique= True, blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.hashed_username:
            self.hashed_username = hashlib.sha256(self.username.encode()).hexdigest()
        super().save(*args, **kwargs)