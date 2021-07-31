from django.db import models

# Create your models here.
class Product(models.Model):
    id = models.AutoField
    Branch_Code = models.CharField(max_length=50)
    Amount = models.IntegerField(default=0)
    NIFTY_PE = models.IntegerField(default=0)
    Type_of_Purchase = models.IntegerField(default=0)

    def __str__(self):
        return self.Branch_Code