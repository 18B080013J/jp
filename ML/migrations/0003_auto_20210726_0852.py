# Generated by Django 3.2.4 on 2021-07-26 03:22

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ML', '0002_auto_20210703_0910'),
    ]

    operations = [
        migrations.RenameField(
            model_name='product',
            old_name='close',
            new_name='Amount',
        ),
        migrations.RenameField(
            model_name='product',
            old_name='product_name',
            new_name='Branch_Code',
        ),
        migrations.RenameField(
            model_name='product',
            old_name='open',
            new_name='NIFTY_PE',
        ),
    ]
