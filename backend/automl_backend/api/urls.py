from django.urls import path
from .views import health_check, preview_csv, run_pipeline, predict_test

urlpatterns = [
    path('health/', health_check, name='health_check'),
    path('preview/', preview_csv, name='preview_csv'),
    path('run/', run_pipeline, name='run_pipeline'),
    path('predict/', predict_test, name='predict_test'),
]
