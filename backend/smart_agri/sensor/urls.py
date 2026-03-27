from django.urls import path
from . import views

urlpatterns = [
    path('add/', views.add_sensor_data),
    path('get/', views.get_sensor_data),
]