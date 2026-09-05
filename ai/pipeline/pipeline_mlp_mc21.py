from ai.pipeline.pipeline_mlp import PipelineMLP
from ai.pipeline.registry import register_pipeline


@register_pipeline("MLP_MC21")
class PipelineMLPmc21(PipelineMLP):
    """
    The ring MLP on the mc21 "isabela QT, 2 sigma restriction" tables.

    Same architecture and same preprocessing as PipelineMLP; the dataset is what differs, and
    it is declared in ai/configs/mlp_mc21.yaml - the rings arrive in one nested list column,
    the label is a real column, and Et/eta sit under their Athena container names.

    It is registered as its own model rather than run through PipelineMLP with another config
    because the results tree is keyed by model name: this way mc21 and mc25 runs land in
    separate directories and appear as separate rows in the cross-validation table, instead of
    silently overwriting each other's regions or being pooled into one set of numbers.

    Anything genuinely mc21-specific - a different ring selection, a different normalisation -
    belongs here as an override of `preprocessor_class`, not as a config switch.
    """
