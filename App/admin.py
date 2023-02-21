from django.contrib import admin
from .models import App
# Register your models here.

class showList(admin.ModelAdmin):
    list_display = ('id','image','result')

admin.site.register(App)
