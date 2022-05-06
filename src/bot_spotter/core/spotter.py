import logging
import os.path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

from bot_spotter.constants import (
    COL_NAMES,
    ENCODER_FILE,
    LABEL,
    MODEL_FILE,
    PREDICTIONS_FILE,
    USER_ID,
)


class Spotter:
    """
    Class representation of the bot detection system
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        random_state: int = 42,
        col_names: str = None,
        label_name: str = None,
        output_dir: str = None,
        val_mode: bool = False,
    ):
        self._random_state = random_state
        self._col_names = COL_NAMES if col_names is None else col_names
        self._label_name = LABEL if label_name is None else label_name
        self._output_dir = output_dir
        self._ohe = (
            self._load(os.path.join(self._output_dir, ENCODER_FILE))
            if val_mode
            else OneHotEncoder(handle_unknown="ignore")
        )
        self.model = (
            self._load(os.path.join(self._output_dir, MODEL_FILE))
            if val_mode
            else LogisticRegression(random_state=self._random_state)
        )

    def _fit_encoder(self, raw_data: pd.DataFrame) -> None:
        self._ohe.fit(raw_data[self._col_names])
        if self._output_dir is not None:
            joblib.dump(self._ohe, os.path.join(self._output_dir, ENCODER_FILE))

    def _build_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        features_array = self._ohe.transform(raw_data[self._col_names]).toarray()
        features_labels = [
            item[j] for item in self._ohe.categories_ for j in range(len(item))
        ]
        data_hot_encoded = pd.DataFrame(
            features_array,
            index=raw_data.index,
            columns=features_labels,
        )
        data_other_cols = raw_data.drop(columns=self._col_names)

        return pd.concat([data_hot_encoded, data_other_cols], axis=1)

    def run(
        self,
        raw_data: pd.DataFrame,
    ) -> None:
        """
        Processes the raw data and fit the model
        """
        self.logger.info("training...")
        self._fit_encoder(raw_data=raw_data)
        features_df = self._build_features(raw_data=raw_data)
        columns_to_drop = [
            col for col in raw_data.columns if col not in self._col_names
        ]
        self.model.fit(
            features_df.drop(columns=columns_to_drop), features_df[self._label_name]
        )

        score = f1_score(
            features_df[self._label_name],
            self.model.predict(features_df.drop(columns=columns_to_drop)),
        )

        msg = f"Training finished. \n Model's f1_score on train: {score}.\n"
        if self._output_dir is not None:
            joblib.dump(self.model, os.path.join(self._output_dir, MODEL_FILE))
            msg += f"Saved artifacts in {self._output_dir}"
        self.logger.info(msg)

    def predict(self, raw_test) -> np.array:
        self.logger.info("Predicting...")
        raw_test.set_index(USER_ID, inplace=True)
        columns_to_drop = [
            col for col in raw_test.columns if col not in self._col_names
        ]

        preds = self.model.predict(
            self._build_features(raw_data=raw_test).drop(columns=columns_to_drop)
        )
        preds = pd.DataFrame(preds, index=raw_test.index, columns=["Prediction"])
        msg = "Predictions done."
        if self._output_dir is not None:
            preds.to_csv(os.path.join(self._output_dir, PREDICTIONS_FILE))
            msg += (
                f"Saved artifacts in {os.path.join(self._output_dir, PREDICTIONS_FILE)}"
            )
        self.logger.info(msg)
        return preds

    @staticmethod
    def _load(path):
        return joblib.load(path)
