from django.urls import path
from .views import (
    CleanlinessReportView,
    ValidateDataView,
    CleanDataView,
    EncodeDataView,
    PreprocessPipelineView,
)

urlpatterns = [
    path("preprocess/cleanliness/", CleanlinessReportView.as_view(), name="pre-cleanliness"),
    path("preprocess/validate/", ValidateDataView.as_view(), name="pre-validate"),
    path("preprocess/clean/", CleanDataView.as_view(), name="pre-clean"),
    path("preprocess/encode/", EncodeDataView.as_view(), name="pre-encode"),
    path("preprocess/pipeline/", 
    PreprocessPipelineView.as_view(),
     name="pre-pipeline"),
]


