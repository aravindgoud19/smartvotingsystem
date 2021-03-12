from django.db import models

# Create your models here.

class user(models.Model):
        username=models.CharField(max_length=20)
        password=models.CharField(max_length=20)
        voterid=models.CharField(max_length=20)
        status=models.BooleanField(default=False)
    
    

class voterid(models.Model):
        First_Name=models.CharField(max_length=20)
        Last_Name=models.CharField(max_length=20)
        voterid=models.CharField(max_length=20,primary_key=True,default=None)
        DOB=models.DateField(auto_now=False,default=None)
        Gender=models.CharField(max_length=8)
        Phoneno=models.CharField(max_length=11)
        Address=models.CharField(max_length=80,default=None)
        locationid=models.CharField(max_length=20,default=None)
        state=models.CharField(max_length=15,default=None)
       


class election(models.Model):
        election_name=models.CharField(max_length=30)
        election_date = models.DateField(default=None)
        locationID=models.CharField(max_length=30)
        class Meta:
                verbose_name_plural="election"
      


class party(models.Model):
        CandidateName=models.CharField(max_length=30)
        PartyName=models.CharField(max_length=40)
        nvotes=models.IntegerField(default=0)
        location=models.CharField(max_length=30,default=None)
        locationID=models.ForeignKey(election,on_delete=models.CASCADE,default=None)
        symbol=models.ImageField(upload_to='pics',blank=True)
        