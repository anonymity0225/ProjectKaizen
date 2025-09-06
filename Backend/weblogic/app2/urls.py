from django.urls import path
from .views import (
    TrainModelView,
    PredictView,
    ListModelsView,
    ModelInfoView,
    DeleteModelView,
    PerformanceComparisonView,
    ExportModelConfigView,
    ValidateModelHealthView,
    CleanupBrokenModelsView,
)

urlpatterns = [
    path("modeling/train", TrainModelView.as_view(), name="model-train"),
    path("modeling/predict", PredictView.as_view(), name="model-predict"),
    path("modeling/list", ListModelsView.as_view(), name="model-list"),
    path("modeling/info/<str:model_id>", ModelInfoView.as_view(), name="model-info"),
    path("modeling/delete/<str:model_id>", DeleteModelView.as_view(), name="model-delete"),
    path("modeling/performance", PerformanceComparisonView.as_view(), name="model-performance"),
    path("modeling/export-config/<str:model_id>", ExportModelConfigView.as_view(), name="model-export-config"),
    path("modeling/health/<str:model_id>", ValidateModelHealthView.as_view(), name="model-health"),
    path("modeling/cleanup", CleanupBrokenModelsView.as_view(), name="model-cleanup"),
]


