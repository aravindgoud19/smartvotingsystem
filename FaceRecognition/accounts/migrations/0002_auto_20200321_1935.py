# Generated by Django 3.0.4 on 2020-03-21 14:05

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='election',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('election_name', models.CharField(max_length=30)),
                ('election_date', models.DateField()),
                ('locationid', models.CharField(max_length=30)),
            ],
        ),
        migrations.AddField(
            model_name='party',
            name='location',
            field=models.CharField(default=None, max_length=3),
        ),
        migrations.AddField(
            model_name='party',
            name='symbol',
            field=models.ImageField(blank=True, upload_to='pics'),
        ),
        migrations.AddField(
            model_name='user',
            name='status',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='voterid',
            name='Address',
            field=models.CharField(default=None, max_length=80),
        ),
        migrations.AddField(
            model_name='voterid',
            name='locationid',
            field=models.CharField(default=None, max_length=20),
        ),
        migrations.AddField(
            model_name='voterid',
            name='state',
            field=models.CharField(default=None, max_length=15),
        ),
        migrations.AlterField(
            model_name='voterid',
            name='voterid',
            field=models.CharField(default=None, max_length=20, primary_key=True, serialize=False),
        ),
        migrations.AddField(
            model_name='party',
            name='locationid',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='accounts.election'),
        ),
    ]
