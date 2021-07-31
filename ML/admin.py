from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from .models import *
# from .models import Product
# admin.site.register(Product)
@admin.register(Product)
class ViewAdmin(ImportExportModelAdmin):
    pass