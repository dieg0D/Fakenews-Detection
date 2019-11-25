# Generated by Django 2.2.7 on 2019-11-25 10:48

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_auto_20191113_0902'),
    ]

    operations = [
        migrations.AddField(
            model_name='formnews',
            name='titulo',
            field=models.CharField(default=django.utils.timezone.now, max_length=200),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='formnews',
            name='veiculo',
            field=models.URLField(default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]
