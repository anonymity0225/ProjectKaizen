import io
import json
import numpy as np
import pandas as pd
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
try:
    from Core.ProjKaizen.app.schemas.preprocess import CleanlinessReport, EncodingConfig, ValidationReport
except Exception:
    CleanlinessReport = None  # type: ignore
    EncodingConfig = None  # type: ignore
    ValidationReport = None  # type: ignore


def _df_from_upload(f):
    name = (f.name or "").lower()
    content = f.read()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(content))
    if name.endswith(".json"):
        return pd.read_json(io.BytesIO(content))
    return None


def _cleanliness_report(df: pd.DataFrame) -> dict:
    total_rows, total_cols = df.shape
    return {
        "total_rows": int(total_rows),
        "total_columns": int(total_cols),
        "missing_per_column": {c: float(df[c].isna().mean()) for c in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
        "column_types": {c: str(t) for c, t in df.dtypes.items()},
        "categorical_cardinality": {c: int(df[c].nunique()) for c in df.select_dtypes(include=["object", "category"]).columns},
    }


def _validate_df(df: pd.DataFrame) -> dict:
    issues = {}
    warnings = []
    if df.empty:
        issues["empty"] = True
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        issues["null_columns"] = null_columns
    return {
        "valid": len(issues) == 0,
        "errors": [],
        "warnings": warnings,
        "details": issues,
    }


def _apply_cleaning(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list]:
    cleaned = df.copy()
    actions = []
    # remove duplicates
    if cfg.get("remove_duplicates", True) or cfg.get("drop_duplicates", True):
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        after = len(cleaned)
        if after != before:
            actions.append({"action": "remove_duplicates", "rows_removed": before - after})

    # missing values
    mv_strategy = cfg.get("missing_value_strategy")
    if mv_strategy is None:
        # map legacy to internal
        m = cfg.get("missing_strategy")
        if m == "mean":
            mv_strategy = "fill_mean"
        elif m == "median":
            mv_strategy = "fill_median"
        elif m == "mode":
            mv_strategy = "fill_mode"
        elif m == "drop":
            mv_strategy = "drop_rows"

    num_cols = cfg.get("numeric_columns") or cleaned.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = cfg.get("categorical_columns") or cleaned.select_dtypes(include=["object", "category"]).columns.tolist()

    if mv_strategy == "fill_mean":
        for c in num_cols:
            if c in cleaned.columns and cleaned[c].isna().any():
                val = cleaned[c].mean()
                cleaned[c] = cleaned[c].fillna(val)
                actions.append({"action": "fillna", "column": c, "method": "mean", "value": float(val)})
    elif mv_strategy == "fill_median":
        for c in num_cols:
            if c in cleaned.columns and cleaned[c].isna().any():
                val = cleaned[c].median()
                cleaned[c] = cleaned[c].fillna(val)
                actions.append({"action": "fillna", "column": c, "method": "median", "value": float(val)})
    elif mv_strategy == "fill_mode":
        for c in cat_cols:
            if c in cleaned.columns and cleaned[c].isna().any():
                mode_vals = cleaned[c].mode()
                if not mode_vals.empty:
                    val = mode_vals.iloc[0]
                    cleaned[c] = cleaned[c].fillna(val)
                    actions.append({"action": "fillna", "column": c, "method": "mode", "value": str(val)})
    elif mv_strategy == "drop_rows":
        before = len(cleaned)
        cleaned = cleaned.dropna()
        after = len(cleaned)
        actions.append({"action": "dropna", "rows_dropped": before - after})

    return cleaned, actions


def _apply_encoding(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list, dict | None]:
    encoded = df.copy()
    actions = []
    encoder_ids = {}
    cat_cols = cfg.get("categorical_columns") or encoded.select_dtypes(include=["object", "category"]).columns.tolist()
    method = cfg.get("method") or cfg.get("categorical_encoding_method") or "label"

    if method == "onehot":
        for c in cat_cols:
            dummies = pd.get_dummies(encoded[c].astype(str), prefix=c)
            encoded = pd.concat([encoded.drop(columns=[c]), dummies], axis=1)
            actions.append({"action": "onehot_encode", "column": c, "columns_created": int(dummies.shape[1])})
    elif method == "label":
        for c in cat_cols:
            le = LabelEncoder()
            encoded[c] = le.fit_transform(encoded[c].astype(str))
            actions.append({"action": "label_encode", "column": c, "classes": le.classes_.tolist()})
    elif method == "target":
        for c in cat_cols:
            vc = encoded[c].value_counts()
            encoded[c] = encoded[c].map(vc).fillna(0)
            actions.append({"action": "target_encode", "column": c})

    # numeric scaling
    scaling = cfg.get("numeric_scaling_method") or cfg.get("scaling_method")
    if scaling and scaling != "none":
        num_cols = cfg.get("numeric_columns") or encoded.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            if scaling == "standard":
                scaler = StandardScaler()
            elif scaling == "minmax":
                scaler = MinMaxScaler()
            elif scaling == "robust":
                scaler = RobustScaler()
            else:
                scaler = None
            if scaler is not None:
                encoded[num_cols] = scaler.fit_transform(encoded[num_cols])
                actions.append({"action": "scale", "method": scaling, "columns": num_cols})

    return encoded, actions, encoder_ids or None


def _safe_preview(df: pd.DataFrame, limit: int = 50) -> list[dict]:
    # Replace infinities with NaN, then convert NaN to None for JSON compliance
    safe = df.replace([np.inf, -np.inf], np.nan).where(pd.notna(df), None)
    return safe.head(limit).to_dict(orient="records")


def _json_safe(value):
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


class CleanlinessReportView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        f = request.FILES.get("file")
        if not f:
            return JsonResponse({"detail": "file is required"}, status=400)
        df = _df_from_upload(f)
        if df is None:
            return JsonResponse({"detail": "Unsupported file type"}, status=400)
        # Use simple local logic to produce a cleanliness dict; DRF browsable API will render Response
        return Response(_json_safe(_cleanliness_report(df)))


class ValidateDataView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        f = request.FILES.get("file")
        if not f:
            return JsonResponse({"detail": "file is required"}, status=400)
        df = _df_from_upload(f)
        if df is None:
            return JsonResponse({"detail": "Unsupported file type"}, status=400)
        return Response(_json_safe(_validate_df(df)))


class CleanDataView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        f = request.FILES.get("file")
        if not f:
            return JsonResponse({"detail": "file is required"}, status=400)
        df = _df_from_upload(f)
        if df is None:
            return JsonResponse({"detail": "Unsupported file type"}, status=400)

        # optional JSON config in a "config" field
        raw = request.POST.get("config")
        cfg = json.loads(raw) if raw else {}

        cleaned, actions = _apply_cleaning(df, cfg or {})
        return Response(_json_safe({
            "original_shape": list(df.shape),
            "final_shape": list(cleaned.shape),
            "actions": actions,
            "rows_removed": int(df.shape[0] - cleaned.shape[0]),
            "columns_removed": int(df.shape[1] - cleaned.shape[1]),
            "preview": _safe_preview(cleaned, 50),
        }))


class EncodeDataView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        f = request.FILES.get("file")
        if not f:
            return JsonResponse({"detail": "file is required"}, status=400)
        df = _df_from_upload(f)
        if df is None:
            return JsonResponse({"detail": "Unsupported file type"}, status=400)

        raw = request.POST.get("config")
        cfg = json.loads(raw) if raw else {}

        encoded, actions, encoder_ids = _apply_encoding(df, cfg or {})
        return Response(_json_safe({
            "original_shape": list(df.shape),
            "final_shape": list(encoded.shape),
            "actions": actions,
            "encoder_ids": encoder_ids,
            "preview": _safe_preview(encoded, 50),
        }))


class PreprocessPipelineView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        f = request.FILES.get("file")
        if not f:
            return JsonResponse({"detail": "file is required"}, status=400)
        df = _df_from_upload(f)
        if df is None:
            return JsonResponse({"detail": "Unsupported file type"}, status=400)

        # configs can be sent as separate fields
        raw_clean = request.POST.get("cleaning")
        raw_encode = request.POST.get("encoding")
        clean_cfg = json.loads(raw_clean) if raw_clean else {}
        enc_cfg = json.loads(raw_encode) if raw_encode else {}

        initial = _cleanliness_report(df)
        validation = _validate_df(df)
        current = df
        actions = []
        if clean_cfg:
            current, clean_actions = _apply_cleaning(current, clean_cfg)
            actions.extend(clean_actions)
        if enc_cfg:
            current, enc_actions, _ = _apply_encoding(current, enc_cfg)
            actions.extend(enc_actions)
        final = _cleanliness_report(current)
        return Response(_json_safe({
            "rows": int(current.shape[0]),
            "columns": int(current.shape[1]),
            "preview": _safe_preview(current, 50),
            "initial_cleanliness": initial,
            "final_cleanliness": final,
            "issues": validation.get("details", {}),
            "actions": actions,
        }))

