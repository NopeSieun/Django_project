from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'App'  
urlpatterns = [
    # two paths: with or without given image
    path('', views.index, name='index'),
    path('imgList.html/',views.ListFunc),
    path('results.html/',views.result_info),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)