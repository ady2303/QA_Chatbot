# Generated by Django 5.0 on 2024-11-22 04:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='document',
            options={'ordering': ['-created_at']},
        ),
        migrations.AlterField(
            model_name='document',
            name='content',
            field=models.TextField(blank=True),
        ),
    ]
