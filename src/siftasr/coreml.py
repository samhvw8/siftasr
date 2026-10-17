"""Core ML models through pyobjc: compile once, load on CPU + GPU, predict with numpy in and out.

coremltools has no Python 3.14 build, so models are run with the CoreML framework directly.
Every prediction runs inside an autorelease pool: Core ML returns autoreleased objects that a
Python loop never drains (QUAN kept ~380 MB per call, 36 GB on a 47 min talk, without it).
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

import numpy as np


def available() -> bool:
    """Apple Silicon, macOS 15+ (multifunction models), pyobjc's CoreML installed."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    major = platform.mac_ver()[0].split(".")[0]
    if not (major.isdigit() and int(major) >= 15):
        return False
    try:
        import CoreML  # noqa: F401
        return True
    except ImportError:
        return False


def load(package: Path, function: str | None = None):
    """Compile `package` (.mlpackage) to a .mlmodelc beside it once, then load it on CPU + GPU."""
    import CoreML
    import Foundation
    compiled = package.with_suffix(".mlmodelc")
    if not compiled.exists():
        tmp, err = CoreML.MLModel.compileModelAtURL_error_(Foundation.NSURL.fileURLWithPath_(str(package)), None)
        if tmp is None:
            raise RuntimeError(f"Core ML could not compile {package}: {err}")
        shutil.move(tmp.path(), compiled)
    cfg = CoreML.MLModelConfiguration.alloc().init()
    cfg.setComputeUnits_(CoreML.MLComputeUnitsCPUAndGPU)
    if function:
        cfg.setFunctionName_(function)
    m, err = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
        Foundation.NSURL.fileURLWithPath_(str(compiled)), cfg, None)
    if m is None:
        raise RuntimeError(f"Core ML could not load {compiled}: {err}")
    return m


def _to_ml(x: np.ndarray):
    import CoreML
    kind = {np.float32: CoreML.MLMultiArrayDataTypeFloat32, np.int32: CoreML.MLMultiArrayDataTypeInt32}[x.dtype.type]
    a, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(list(x.shape), kind, None)
    if a is None:
        raise RuntimeError(f"MLMultiArray: {err}")
    np.frombuffer(a.dataPointer().as_buffer(x.nbytes), x.dtype)[:] = np.ascontiguousarray(x).ravel()
    return a


def _to_np(ma) -> np.ndarray:
    import CoreML
    if ma.dataType() != CoreML.MLMultiArrayDataTypeFloat32:
        raise RuntimeError(f"unexpected Core ML output type {ma.dataType()}")
    shape, strides = [int(s) for s in ma.shape()], [int(s) for s in ma.strides()]
    buf = np.frombuffer(ma.dataPointer().as_buffer(strides[0] * shape[0] * 4), np.float32)
    return np.lib.stride_tricks.as_strided(buf, shape, [s * 4 for s in strides]).copy()


def predict(model, inputs: dict[str, np.ndarray], outputs: list[str]) -> list[np.ndarray]:
    """One prediction: float32 / int32 arrays in, float32 arrays out (copied out of Core ML's buffers)."""
    import CoreML
    import objc
    with objc.autorelease_pool():
        feats = {k: CoreML.MLFeatureValue.featureValueWithMultiArray_(_to_ml(v)) for k, v in inputs.items()}
        prov, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(feats, None)
        out, err = model.predictionFromFeatures_error_(prov, None) if prov is not None else (None, err)
        if out is None:
            raise RuntimeError(f"Core ML prediction failed: {err}")
        return [_to_np(out.featureValueForName_(o).multiArrayValue()) for o in outputs]
