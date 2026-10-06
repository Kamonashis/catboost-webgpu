"""Dataset container Pool for catboost_webgpu."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Union
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

from catboost_webgpu._core import PyPool


class Pool:
    """Pool dataset holding features, labels, weights, group IDs, and categorical metadata.

    Supports Pandas DataFrames, NumPy ndarrays, and Python nested sequences.
    """

    def __init__(
        self,
        data: Any,
        label: Optional[Any] = None,
        cat_features: Optional[Sequence[Union[int, str]]] = None,
        text_features: Optional[Sequence[Union[int, str]]] = None,
        weight: Optional[Any] = None,
        group_id: Optional[Any] = None,
        group_weight: Optional[Any] = None,
        subgroup_id: Optional[Any] = None,
        pairs: Optional[Any] = None,
        baseline: Optional[Any] = None,
        feature_names: Optional[Sequence[str]] = None,
        thread_count: int = -1,
        max_borders: int = 254,
        quantization_method: str = "median",
        one_hot_max_size: int = 64,
        seed: int = 42,
        cat_mappings: Optional[Dict[int, Dict[Any, int]]] = None,
        **kwargs: Any,
    ) -> None:
        self._one_hot_max_size = one_hot_max_size
        self._cat_mappings: Dict[int, Dict[Any, int]] = (
            {int(k): dict(v) for k, v in cat_mappings.items()} if cat_mappings is not None else {}
        )
        if isinstance(data, Pool):
            # Shallow clone
            self._pool = data._pool
            self._data = data._data
            self._label = data._label
            self._weight = data._weight
            self._group_id = data._group_id
            self._cat_features = list(data._cat_features)
            self._feature_names = list(data._feature_names)
            self._cat_mappings = dict(data._cat_mappings)
            self._one_hot_max_size = data._one_hot_max_size
            return

        if isinstance(data, PyPool):
            self._pool = data
            self._data = None
            self._label = np.asarray(data.get_targets()) if data.has_target else None
            self._weight = np.asarray(data.get_weights()) if data.get_weights() is not None else None
            self._group_id = np.asarray(data.get_group_ids()) if data.get_group_ids() is not None else None
            self._cat_features = []
            self._feature_names = data.get_feature_names()
            return

        # 1. Parse feature names and detect categorical features
        parsed_feature_names: Optional[List[str]] = None
        if feature_names is not None:
            parsed_feature_names = [str(name) for name in feature_names]

        cat_feature_indices: List[int] = []
        df_columns: Optional[List[str]] = None

        # Check if Pandas DataFrame
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            df_columns = [str(c) for c in data.columns]
            if parsed_feature_names is None:
                parsed_feature_names = list(df_columns)

            # Auto-detect category or string columns
            for idx, col in enumerate(data.columns):
                dtype = data[col].dtype
                if dtype.name == "category" or dtype == object or dtype == "string":
                    cat_feature_indices.append(idx)

            # Process explicit cat_features argument
            if cat_features is not None:
                for cf in cat_features:
                    if isinstance(cf, str):
                        if cf in df_columns:
                            c_idx = df_columns.index(cf)
                            if c_idx not in cat_feature_indices:
                                cat_feature_indices.append(c_idx)
                        else:
                            raise ValueError(f"Categorical feature '{cf}' not found in DataFrame columns")
                    else:
                        c_idx = int(cf)
                        if c_idx not in cat_feature_indices:
                            cat_feature_indices.append(c_idx)

            # Convert DataFrame columns to numeric representation
            cat_feature_indices.sort()
            processed_cols = []
            for idx, col in enumerate(data.columns):
                s = data[col]
                if idx in cat_feature_indices:
                    str_vals = [str(val) for val in s]
                    if idx in self._cat_mappings:
                        mapping = self._cat_mappings[idx]
                        codes = np.array([float(mapping.get(val, 0)) for val in str_vals], dtype=np.float32)
                    else:
                        uniques = pd.unique(str_vals)
                        mapping = {val: i for i, val in enumerate(uniques)}
                        self._cat_mappings[idx] = mapping
                        codes = np.array([float(mapping[val]) for val in str_vals], dtype=np.float32)
                    processed_cols.append(codes)
                else:
                    processed_cols.append(pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))

            X_mat = np.column_stack(processed_cols).astype(np.float32, copy=False)

        elif hasattr(data, "__array__") or isinstance(data, (list, tuple, np.ndarray)):
            # Convert to numpy array
            arr = np.asarray(data)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            if cat_features is not None:
                for cf in cat_features:
                    if isinstance(cf, int):
                        cat_feature_indices.append(cf)
                    elif isinstance(cf, str) and parsed_feature_names is not None:
                        if cf in parsed_feature_names:
                            cat_feature_indices.append(parsed_feature_names.index(cf))
                        else:
                            raise ValueError(f"Feature name '{cf}' not in feature_names")
                    else:
                        raise ValueError(f"Invalid categorical feature specification: {cf}")

            cat_feature_indices.sort()

            # Handle object arrays containing strings/categories
            if arr.dtype == object or np.issubdtype(arr.dtype, np.str_):
                num_cols = arr.shape[1]
                processed_cols = []
                for c in range(num_cols):
                    col_vals = arr[:, c]
                    if c in cat_feature_indices:
                        str_vals = [str(val) for val in col_vals]
                        if c in self._cat_mappings:
                            mapping = self._cat_mappings[c]
                            codes = np.array([float(mapping.get(val, 0)) for val in str_vals], dtype=np.float32)
                        else:
                            uniques = np.unique(str_vals)
                            mapping = {val: i for i, val in enumerate(uniques)}
                            self._cat_mappings[c] = mapping
                            codes = np.array([float(mapping[val]) for val in str_vals], dtype=np.float32)
                        processed_cols.append(codes)
                    else:
                        try:
                            processed_cols.append(col_vals.astype(np.float32))
                        except (ValueError, TypeError):
                            # Fallback if conversion fails
                            if HAS_PANDAS:
                                codes, _ = pd.factorize(col_vals)
                                codes = codes.astype(np.float32)
                            else:
                                _, codes = np.unique(col_vals.astype(str), return_inverse=True)
                                codes = codes.astype(np.float32)
                            if c not in cat_feature_indices:
                                cat_feature_indices.append(c)
                            processed_cols.append(codes)
                X_mat = np.column_stack(processed_cols).astype(np.float32, copy=False)
                cat_feature_indices.sort()
            else:
                X_mat = np.ascontiguousarray(arr, dtype=np.float32)

            if parsed_feature_names is None:
                parsed_feature_names = [f"feature_{i}" for i in range(X_mat.shape[1])]
        else:
            raise TypeError(f"Unsupported data type for Pool: {type(data)}")

        # 2. Parse label
        y_vec: Optional[np.ndarray] = None
        if label is not None:
            if HAS_PANDAS and isinstance(label, (pd.Series, pd.DataFrame)):
                y_vec = label.to_numpy(dtype=np.float32).ravel()
            else:
                y_vec = np.asarray(label, dtype=np.float32).ravel()
            if len(y_vec) != X_mat.shape[0]:
                raise ValueError(
                    f"Label length {len(y_vec)} does not match number of samples {X_mat.shape[0]}"
                )

        # 3. Parse weights
        w_vec: Optional[np.ndarray] = None
        if weight is not None:
            if HAS_PANDAS and isinstance(weight, (pd.Series, pd.DataFrame)):
                w_vec = weight.to_numpy(dtype=np.float32).ravel()
            else:
                w_vec = np.asarray(weight, dtype=np.float32).ravel()

        # 4. Parse group_id
        g_vec: Optional[np.ndarray] = None
        if group_id is not None:
            if HAS_PANDAS and isinstance(group_id, (pd.Series, pd.DataFrame)):
                g_vals = group_id.to_numpy()
            else:
                g_vals = np.asarray(group_id)
            if g_vals.dtype == object or np.issubdtype(g_vals.dtype, np.str_):
                if HAS_PANDAS:
                    codes, _ = pd.factorize(g_vals.ravel())
                    g_vec = codes.astype(np.uint32)
                else:
                    _, codes = np.unique(g_vals.ravel(), return_inverse=True)
                    g_vec = codes.astype(np.uint32)
            else:
                g_vec = np.asarray(g_vals, dtype=np.uint32).ravel()

        # Build PyPool native extension object
        self._pool = PyPool(
            data=X_mat,
            targets=y_vec,
            cat_features=cat_feature_indices if len(cat_feature_indices) > 0 else None,
            weights=w_vec,
            group_ids=g_vec,
            feature_names=parsed_feature_names,
            max_borders=max_borders,
            quantization_method=quantization_method,
            one_hot_max_size=one_hot_max_size,
            seed=seed,
        )

        self._data = X_mat
        self._label = y_vec
        self._weight = w_vec
        self._group_id = g_vec
        self._cat_features = cat_feature_indices
        self._feature_names = parsed_feature_names or [f"feature_{i}" for i in range(X_mat.shape[1])]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_row(), self.num_col())

    def num_row(self) -> int:
        return self._pool.num_samples

    def num_col(self) -> int:
        return len(self._feature_names) if self._feature_names else self._pool.num_features

    def get_features(self) -> Optional[np.ndarray]:
        if self._data is not None:
            return self._data
        raw = self._pool.get_raw_features()
        if raw is not None:
            return np.asarray(raw, dtype=np.float32).reshape(self.shape)
        return None

    def get_label(self) -> Optional[np.ndarray]:
        if self._pool.has_target:
            return np.asarray(self._pool.get_targets(), dtype=np.float32)
        return None

    def get_weight(self) -> Optional[np.ndarray]:
        w = self._pool.get_weights()
        return np.asarray(w, dtype=np.float32) if w is not None else None

    def get_group_id(self) -> Optional[np.ndarray]:
        g = self._pool.get_group_ids()
        return np.asarray(g, dtype=np.uint32) if g is not None else None

    @property
    def cat_features(self) -> List[int]:
        return list(self._cat_features)

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def cat_mappings(self) -> Dict[int, Dict[Any, int]]:
        return dict(self._cat_mappings)

    @property
    def one_hot_max_size(self) -> int:
        return self._one_hot_max_size

    def slice(self, rindices: Union[Sequence[int], range, slice]) -> Pool:
        if isinstance(rindices, slice):
            start = rindices.start or 0
            stop = rindices.stop or self.num_row()
            sliced_pypool = self._pool.slice(start, stop)
            new_p = Pool.__new__(Pool)
            new_p._pool = sliced_pypool
            new_p._data = self._data[start:stop] if self._data is not None else None
            new_p._label = self._label[start:stop] if self._label is not None else None
            new_p._weight = self._weight[start:stop] if self._weight is not None else None
            new_p._group_id = self._group_id[start:stop] if self._group_id is not None else None
            new_p._cat_features = list(self._cat_features)
            new_p._feature_names = list(self._feature_names)
            return new_p
        elif isinstance(rindices, range):
            return self.slice(slice(rindices.start, rindices.stop, rindices.step))
        else:
            # Sequence of indices
            indices = list(rindices)
            if self._data is not None:
                sub_data = self._data[indices]
                sub_label = self._label[indices] if self._label is not None else None
                sub_weight = self._weight[indices] if self._weight is not None else None
                sub_group = self._group_id[indices] if self._group_id is not None else None
                return Pool(
                    data=sub_data,
                    label=sub_label,
                    cat_features=self._cat_features,
                    weight=sub_weight,
                    group_id=sub_group,
                    feature_names=self._feature_names,
                )
            else:
                raise NotImplementedError("Arbitrary index slicing on raw PyPool without cached data")

    def __len__(self) -> int:
        return self._pool.num_samples

    def __repr__(self) -> str:
        return (
            f"<catboost_webgpu.Pool samples={self.num_row()} "
            f"features={self.num_col()} "
            f"cat_features={len(self._cat_features)} "
            f"has_target={self._pool.has_target}>"
        )
