from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_scan, name='upload_scan'),
    path('result/<int:pk>/', views.scan_result, name='scan_result'),
]