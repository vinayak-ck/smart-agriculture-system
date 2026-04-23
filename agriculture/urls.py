from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/sensor-data', views.api_sensor_data, name='api_sensor_data'),
    path('api/get-data', views.api_get_data, name='api_get_data'),
    path('api/predict-growth', views.api_predict_growth, name='api_predict_growth'),
    path('api/predict-disease', views.api_predict_disease, name='api_predict_disease'),
    path('api/recent-images', views.api_recent_images, name='api_recent_images'),
    path('api/growth-chart', views.api_growth_chart, name='api_growth_chart'),
    path('api/new-session', views.api_new_session, name='api_new_session'),
]
