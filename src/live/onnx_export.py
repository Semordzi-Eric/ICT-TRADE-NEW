"""Export the trained LightGBM model to ONNX so MQL5 / ONNX Runtime can run it.

The MQL5 ``OnnxRun`` API accepts a single .onnx file. Below the export we
include a commented block showing the corresponding MQL5 inference snippet.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def export_lightgbm_to_onnx(
    booster,
    feature_columns: List[str],
    output_path: str = "models_artifacts/lightgbm.onnx",
) -> str:
    """Convert a LightGBM Booster to ONNX. Returns the saved file path."""
    try:
        from onnxmltools import convert_lightgbm
        from onnxconverter_common.data_types import FloatTensorType
    except ImportError as exc:
        raise ImportError(
            "onnxmltools and onnxconverter-common are required for LightGBM ONNX export"
        ) from exc

    n_features = len(feature_columns)
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(booster, initial_types=initial_types, target_opset=15)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("LightGBM exported to %s", out)
    return str(out)


def export_sklearn_to_onnx(
    model,
    feature_columns: List[str],
    output_path: str,
) -> str:
    """Export an sklearn meta-model (e.g. logistic regression) to ONNX."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise ImportError("skl2onnx is required for sklearn ONNX export") from exc

    initial_types = [("input", FloatTensorType([None, len(feature_columns)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=15)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("Sklearn model exported to %s", out)
    return str(out)


def verify_onnx(onnx_path: str, X_sample: np.ndarray) -> np.ndarray:
    """Run the exported ONNX model with onnxruntime to sanity-check predictions."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is required to verify ONNX models") from exc
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: X_sample.astype(np.float32)})
    return out[0]


# ----------------------------------------------------------------------------
# MQL5 reference snippet — paste into an Expert Advisor and load the .onnx file:
#
#   #resource "lightgbm.onnx" as uchar ext_model[]
#
#   long handle = OnnxCreateFromBuffer(ext_model, ONNX_DEFAULT);
#   ulong input_shape[]  = {1, 35};
#   ulong output_shape[] = {1, 1};
#   OnnxSetInputShape(handle, 0, input_shape);
#   OnnxSetOutputShape(handle, 0, output_shape);
#
#   float input_data[35];
#   // ... fill input_data with the same 35 features used during training ...
#   float output_data[1];
#
#   if(OnnxRun(handle, ONNX_DEFAULT, input_data, output_data))
#   {
#       double prob = (double)output_data[0];
#       if(prob >= 0.65)
#           // place trade
#   }
#   OnnxRelease(handle);
# ----------------------------------------------------------------------------
