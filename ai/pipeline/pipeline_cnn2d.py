from ai.pipeline.base import BasePipeline
from ai.pipeline.registry import register_pipeline
from ai.preprocess.cnn2d import PreprocessCNN2D
from ai.models.cnn2d import ModelCNN2D


@register_pipeline("CNN2D")
class PipelineCNN2D(BasePipeline):
    """
    Training and evaluation pipeline for the 2D CNN over calorimeter cell images.

    The input shape is fixed by the preprocessor's padding target (7 layers of 7x15 cells),
    so the architecture's defaults already match and no model kwargs are derived from the data.
    """

    model_class = ModelCNN2D
    preprocessor_class = PreprocessCNN2D
