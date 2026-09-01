import logging
from typing import List

from ai.pipeline.base import BasePipeline
from ai.preprocess.cnn2d import PreprocessCNN2D
from ai.models.cnn2d import ModelCNN2D

logger = logging.getLogger(__name__)


class PipelineCNN2D(BasePipeline):
    """
    Training and evaluation pipeline for the 2D CNN over calorimeter cell images.

    Everything structural lives in BasePipeline; this class only declares which model and
    preprocessor to use. The CNN's input shape is fixed by the architecture (7 layers of
    7x15 padded cells), so no model kwargs are derived from the data.
    """

    model_class = ModelCNN2D

    def build_preprocessor(self) -> PreprocessCNN2D:
        """
        Builds the calorimeter-cell-to-image preprocessor.

        Returns:
            PreprocessCNN2D: A fresh preprocessor.
        """
        return PreprocessCNN2D()

    def required_columns(self, available: List[str]) -> List[str]:
        """
        Restricts the parquet scan to the 7 calorimeter cell-image columns, leaving the 300
        ring/shower-shape columns unread.

        Args:
            available (List[str]): Column names present in the dataset files.

        Returns:
            List[str]: The cell columns the image tensors are built from.
        """
        return list(self.preprocessor.cell_columns)
