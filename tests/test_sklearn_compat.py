import numpy as np
from sklearn.impute import SimpleImputer

from house_price_predictor.sklearn_compat import patch_sklearn_simple_imputers


def test_patch_adds_fill_dtype_on_simple_imputer() -> None:
    # Older pickles may omit _fill_dtype; patch should set it from statistics_ so newer sklearn can transform.
    imp = SimpleImputer(strategy="median")
    imp.fit(np.array([[1.0], [2.0], [np.nan]]))
    if hasattr(imp, "_fill_dtype"):
        delattr(imp, "_fill_dtype")
    patch_sklearn_simple_imputers(imp)
    assert hasattr(imp, "_fill_dtype")
