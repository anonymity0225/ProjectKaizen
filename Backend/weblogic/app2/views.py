import json
from typing import Any

import numpy as np
import pandas as pd
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.views import APIView

try:
    # Prefer using Core modeling service functions directly
    from Core.ProjKaizen.app.services.modeling import (
        train_model as core_train_model,
        predict as core_predict,
        list_models as core_list_models,
        get_model_info as core_get_model_info,
        delete_model as core_delete_model,
        get_model_performance_comparison as core_get_performance,
        export_model_config as core_export_model_config,
        cleanup_broken_models as core_cleanup_broken_models,
        validate_model_health as core_validate_model_health,
    )
    from Core.ProjKaizen.app.schemas.modeling import ModelConfig
except Exception:  # pragma: no cover
    core_train_model = None  # type: ignore
    core_predict = None  # type: ignore
    core_list_models = None  # type: ignore
    core_get_model_info = None  # type: ignore
    core_delete_model = None  # type: ignore
    core_get_performance = None  # type: ignore
    core_export_model_config = None  # type: ignore
    core_cleanup_broken_models = None  # type: ignore
    core_validate_model_health = None  # type: ignore
    ModelConfig = None  # type: ignore


def _json_safe(value: Any):
    """Recursively convert NaN/Inf to None in lists/dicts for JSON safety."""
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_json_safe(list(value)))
    return value


def _model_to_dict(obj: Any) -> dict:
    """Best-effort conversion of pydantic/BaseModel-like objects to dict."""
    if obj is None:
        return {}
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()
        except Exception:
            pass
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj
    # Fallback: try to serialize known attributes
    try:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {"value": str(obj)}


class TrainModelView(APIView):
    def post(self, request):
        if core_train_model is None or ModelConfig is None:
            return JsonResponse({"detail": "Modeling service not available. Ensure Core.ProjKaizen is on PYTHONPATH."}, status=500)

        payload = request.data or {}
        data = payload.get("data")
        config_dict = payload.get("config") or {}

        if not isinstance(data, list) or not data:
            return JsonResponse({"detail": "'data' must be a non-empty list of records"}, status=400)
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            return JsonResponse({"detail": f"Failed to build DataFrame: {e}"}, status=400)

        try:
            config = ModelConfig(**config_dict)
        except Exception as e:
            return JsonResponse({"detail": f"Invalid config: {e}"}, status=422)

        try:
            result = core_train_model(config)
        except Exception as e:
            return JsonResponse({"detail": str(e)}, status=getattr(e, "status_code", 500))

        return Response(_json_safe(_model_to_dict(result)))


class PredictView(APIView):
    def post(self, request):
        if core_predict is None:
            return JsonResponse({"detail": "Modeling service not available. Ensure Core.ProjKaizen is on PYTHONPATH."}, status=500)

        payload = request.data or {}
        model_id = payload.get("model_id")
        data = payload.get("data")

        if not model_id:
            return JsonResponse({"detail": "'model_id' is required"}, status=400)
        if not isinstance(data, list) or not data:
            return JsonResponse({"detail": "'data' must be a non-empty list of records"}, status=400)

        try:
            df = pd.DataFrame(data)
        except Exception as e:
            return JsonResponse({"detail": f"Failed to build DataFrame: {e}"}, status=400)

        try:
            result = core_predict(model_id, df)
        except Exception as e:
            return JsonResponse({"detail": str(e)}, status=getattr(e, "status_code", 500))

        return Response(_json_safe(_model_to_dict(result)))


class ListModelsView(APIView):
    def get(self, request):
        if core_list_models is None:
            return JsonResponse({"detail": "Modeling service not available."}, status=500)
        try:
            result = core_list_models(include_stats=True)
        except Exception as e:
            return JsonResponse({"detail": str(e)}, status=getattr(e, "status_code", 500))
        return Response(_json_safe(_model_to_dict(result)))


class ModelInfoView(APIView):
    def get(self, request, model_id: str):
        if core_get_model_info is None:
            return JsonResponse({"detail": "Modeling service not available."}, status=500)
        try:
            result = core_get_model_info(model_id)
        except Exception as e:
            return JsonResponse({"detail": str(e)}, status=getattr(e, "status_code", 500))
        return Response(_json_safe(_model_to_dict(result)))


class DeleteModelView(APIView):
    def delete(self, request, model_id: str):
        if core_delete_model is None:
            return JsonResponse({"detail": "Modeling service not available."}, status=500)
        try:
            result = core_delete_model(model_id)
        except Exception as e:
            return JsonResponse({"detail": str(e)}, status=getattr(e, "status_code", 500))
        return Response(_json_safe(result))


class PerformanceComparisonView(APIView):
    def get(self, request):
        if core_get_performance is None:
            return JsonResponse({"detail": "Modeling service not available."}, status=500)
        try:
            result = core_get_performance()
        except Exception as e:
            return JsonResponse({"detail": str(e)}, status=getattr(e, "status_code", 500))
        return Response(_json_safe(result))


class ExportModelConfigView(APIView):
    def get(self, request, model_id: str):
        if core_export_model_config is None:
            return JsonResponse({"detail": "Modeling service not available."}, status=500)
        try:
            result = core_export_model_config(model_id)
        except Exception as e:
            return JsonResponse({"detail": str(e)}, status=getattr(e, "status_code", 500))
        return Response(_json_safe(_model_to_dict(result)))


class ValidateModelHealthView(APIView):
    def get(self, request, model_id: str):
        if core_validate_model_health is None:
            return JsonResponse({"detail": "Modeling service not available."}, status=500)
        try:
            result = core_validate_model_health(model_id)
        except Exception as e:
            return JsonResponse({"detail": str(e)}, status=getattr(e, "status_code", 500))
        return Response(_json_safe(_model_to_dict(result)))


class CleanupBrokenModelsView(APIView):
    def post(self, request):
        if core_cleanup_broken_models is None:
            return JsonResponse({"detail": "Modeling service not available."}, status=500)
        try:
            result = core_cleanup_broken_models()
        except Exception as e:
            return JsonResponse({"detail": str(e)}, status=getattr(e, "status_code", 500))
        return Response(_json_safe(_model_to_dict(result)))

