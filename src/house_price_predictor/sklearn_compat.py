from __future__ import annotations

from typing import Any

import numpy as np


def patch_sklearn_simple_imputers(estimator: Any) -> None:
    # Backward compatibility: scikit-learn 1.8+ SimpleImputer.transform uses _fill_dtype; older pickles lack it.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    def visit(obj: Any) -> None:
        if isinstance(obj, SimpleImputer):
            if hasattr(obj, "_fill_dtype"):
                return
            stats = getattr(obj, "statistics_", None)
            if stats is not None and hasattr(stats, "dtype"):
                obj._fill_dtype = stats.dtype
            else:
                obj._fill_dtype = np.dtype("float64")
            return

        if isinstance(obj, Pipeline):
            for _, step in obj.steps:
                if step not in (None, "passthrough"):
                    visit(step)
            return

        if isinstance(obj, ColumnTransformer):
            transformers = getattr(obj, "transformers_", None) or obj.transformers
            for item in transformers:
                trans = item[1]
                if trans not in (None, "drop", "passthrough"):
                    visit(trans)
            remainder = getattr(obj, "remainder_", None) or getattr(obj, "remainder", "drop")
            if remainder not in (None, "drop", "passthrough") and not isinstance(remainder, str):
                visit(remainder)

    visit(estimator)
