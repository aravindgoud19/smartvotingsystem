# Generated by Django 3.0.4 on 2020-03-21 14:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0005_remove_party_img'),
    ]

    operations = [
        migrations.AlterField(
            model_name='party',
            name='location',
            field=models.CharField(default=None, max_length=30),
        ),
    ]