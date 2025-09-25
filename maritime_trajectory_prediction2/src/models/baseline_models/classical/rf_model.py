"""
Random Forest regression baseline with feature importance analysis.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import ClassicalMLBaseline, ClassicalMLConfig


class RFBaseline(ClassicalMLBaseline):
    """
    Random Forest baseline for trajectory prediction.

    Provides feature importance analysis and out-of-bag error estimation.
    Naturally handles multi-output without wrapper.
    """

    def __init__(
        self,
        config: ClassicalMLConfig = None,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        oob_score: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize Random Forest baseline.

        Args:
            config: ClassicalMLConfig instance
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split node
            min_samples_leaf: Minimum samples required at leaf
            max_features: Number of features to consider for best split
            oob_score: Whether to compute out-of-bag score
            random_state: Random seed for reproducibility
        """
        super().__init__(config or ClassicalMLConfig(model_type="rf"))
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.random_state = random_state

        # Store feature importance
        self.feature_importances_ = {}

    def _create_base_model(self):
        """Create Random Forest with maritime-optimized parameters."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            oob_score=self.oob_score,
            n_jobs=self.config.n_jobs,
            random_state=self.random_state,
            warm_start=False,  # Fresh training each time
            verbose=0,
        )

    def _create_horizon_model(self):
        """RF naturally handles multi-output, no wrapper needed."""
        return self._create_base_model()

    def fit(
        self,
        sequences: list[np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> "RFBaseline":
        """
        Fit RF model and extract feature importances.
        """
        # Call parent fit
        super().fit(sequences, metadata, **kwargs)

        # Extract and store feature importances for each horizon
        for horizon, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                self.feature_importances_[horizon] = model.feature_importances_

        return self

    def _estimate_uncertainty(
        self, sequence: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """
        Estimate prediction uncertainty using tree variance.

        RF can provide uncertainty through prediction variance across trees.
        """
        horizon = len(predictions)
        uncertainty = np.zeros_like(predictions)

        X = self._sequence_to_features(sequence)
        X_scaled = self.scaler.transform(X.reshape(1, -1))

        for h in range(1, horizon + 1):
            if h in self.models:
                model = self.models[h]

                # Get predictions from all trees
                tree_predictions = []
                for tree in model.estimators_:
                    tree_pred = tree.predict(X_scaled)
                    tree_predictions.append(tree_pred)

                tree_predictions = np.array(tree_predictions)

                # Compute standard deviation across trees
                if tree_predictions.shape[0] > 1:
                    # Tree predictions shape: [n_trees, 1, n_features]
                    std_per_feature = np.std(tree_predictions, axis=0)[0]
                    uncertainty[h - 1] = std_per_feature
                else:
                    # Fallback uncertainty
                    uncertainty[h - 1] = 0.1 * h * np.ones(predictions.shape[1])

        return uncertainty

    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Generate feature importance report across all horizons.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.feature_importances_:
            raise ValueError("Model must be fitted first")

        # Create feature names based on engineering configuration
        feature_names = self._generate_feature_names()

        # Aggregate importances across horizons
        importance_data = []

        for horizon, importances in self.feature_importances_.items():
            for idx, importance in enumerate(importances):
                importance_data.append(
                    {
                        "horizon": horizon,
                        "feature": (
                            feature_names[idx]
                            if idx < len(feature_names)
                            else f"feature_{idx}"
                        ),
                        "importance": importance,
                    }
                )

        df = pd.DataFrame(importance_data)

        # Compute mean importance across horizons
        mean_importance = (
            df.groupby("feature")["importance"].mean().sort_values(ascending=False)
        )

        return mean_importance

    def _generate_feature_names(self) -> list[str]:
        """Generate human-readable feature names."""
        names = []

        # Assuming 4 base features: [lat, lon, sog, cog]
        base_features = ["lat", "lon", "sog", "cog"]

        # Current features
        names.extend([f"current_{f}" for f in base_features])

        # Lag features
        if self.config.use_lag_features:
            for lag in self.config.lag_steps:
                names.extend([f"lag{lag}_{f}" for f in base_features])

        # Difference features
        if self.config.use_diff_features:
            names.extend([f"diff_{f}" for f in base_features])

        # Rolling features
        if self.config.use_rolling_features:
            for window in self.config.rolling_windows:
                names.extend([f"roll{window}_mean_{f}" for f in base_features])
                names.extend([f"roll{window}_std_{f}" for f in base_features])

        return names

    def get_oob_score(self) -> dict[int, float]:
        """
        Get out-of-bag scores for each horizon model.

        Returns:
            Dict mapping horizon to OOB R² score
        """
        if not self.oob_score:
            raise ValueError("OOB scoring not enabled")

        scores = {}
        for horizon, model in self.models.items():
            if hasattr(model, "oob_score_"):
                scores[horizon] = model.oob_score_

        return scores

    def get_tree_depths(self) -> dict[int, list[int]]:
        """
        Get actual depths of trees in the forest.

        Returns:
            Dict mapping horizon to list of tree depths
        """
        depths = {}

        for horizon, model in self.models.items():
            if hasattr(model, "estimators_"):
                tree_depths = []
                for tree in model.estimators_:
                    # Calculate actual tree depth
                    depth = self._get_tree_depth(tree.tree_)
                    tree_depths.append(depth)
                depths[horizon] = tree_depths

        return depths

    def _get_tree_depth(self, tree):
        """Calculate the actual depth of a decision tree."""
        LEAF_NODE_INDICATOR = -2

        def depth_helper(node):
            if tree.feature[node] == LEAF_NODE_INDICATOR:  # Leaf node
                return 0
            left_depth = depth_helper(tree.children_left[node])
            right_depth = depth_helper(tree.children_right[node])
            return 1 + max(left_depth, right_depth)

        return depth_helper(0)

    def get_model_summary(self) -> dict[str, Any]:
        """
        Get comprehensive model summary including feature importance.

        Returns:
            Dict with model statistics and top features
        """
        summary = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "n_horizons": len(self.models),
            "is_fitted": self.is_fitted,
        }

        if self.is_fitted:
            # Add OOB scores if available
            if self.oob_score:
                try:
                    oob_scores = self.get_oob_score()
                    summary["mean_oob_score"] = np.mean(list(oob_scores.values()))
                    summary["oob_scores_by_horizon"] = oob_scores
                except Exception:  # Specific exception catch
                    pass

            # Add top features
            try:
                importance_df = self.get_feature_importance_report()
                summary["top_5_features"] = importance_df.head(5).to_dict()
            except Exception:  # Specific exception catch
                pass

            # Add tree depth statistics
            tree_depths = self.get_tree_depths()
            if tree_depths:
                all_depths = []
                for depths in tree_depths.values():
                    all_depths.extend(depths)
                summary["mean_tree_depth"] = np.mean(all_depths)
                summary["max_tree_depth"] = np.max(all_depths)

        return summary
