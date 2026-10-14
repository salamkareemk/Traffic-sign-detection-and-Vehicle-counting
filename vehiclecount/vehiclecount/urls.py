
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('countpage', views.countpage, name='countpage'),
    path('countpage2', views.countpage2, name='countpage2'),
    path('video/', views.video, name='video'),
    path('countpagecheck', views.countpagecheck, name='countpagecheck'),
]
