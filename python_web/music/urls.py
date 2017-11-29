from django.conf.urls import url
from . import  views

urlpatterns = [
    url(r'^$',views.index,name='index'),#index is the name of function which is in views.py in music directory
]