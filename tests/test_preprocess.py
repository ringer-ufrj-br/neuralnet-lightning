import numpy as np
import polars as pl

from neuralnet_lightning.data.loader import load_dataframe
from neuralnet_lightning.models.registry import available_models, get_model
from neuralnet_lightning.preprocess.cnn2d import PreprocessCNN2D
from neuralnet_lightning.preprocess.fused import PreprocessFused
from neuralnet_lightning.preprocess.mlp import PreprocessMLP, selected_rings


def test_selected_rings_are_the_reference_selection():
    rings = [int(name.split("_")[1]) for name in selected_rings()]
    assert len(rings) == 50
    # PS 0-3, EM1 8-39, EM2 72-75, EM3 80-83, HAD1 88-89, HAD2 92-93, HAD3 96-97.
    expected = [*range(0, 4), *range(8, 40), *range(72, 76), *range(80, 84), 88, 89, 92, 93, 96, 97]
    assert rings == expected


def test_rings_shape_and_norm1(per_ring_schema):
    preprocessor = PreprocessMLP()
    df = load_dataframe(per_ring_schema, preprocessor.required_columns())
    X = preprocessor.fit_transform(df)

    assert X.shape == (df.height, 50) and X.dtype == np.float32
    np.testing.assert_allclose(np.abs(X.sum(axis=1)), 1.0, rtol=1e-5)
    assert preprocessor.model_kwargs(X) == {"input_dim": 50}


def test_sensor_anomalies_are_zeroed():
    df = pl.DataFrame({name: [1.0, 1.0] for name in selected_rings()})
    df = df.with_columns(pl.Series("ring_0", [-999.0, float("nan")]))
    X = PreprocessMLP().transform(df)
    assert X[:, 0].tolist() == [0.0, 0.0]
    assert np.isfinite(X).all()


def test_cnn2d_images(per_ring_schema):
    preprocessor = PreprocessCNN2D()
    df = load_dataframe(per_ring_schema, preprocessor.required_columns())
    X = preprocessor.fit_transform(df)

    assert X.shape == (df.height, 7, 7, 15) and X.dtype == np.float32
    np.testing.assert_allclose(X.sum(axis=(1, 2, 3)), 1.0, rtol=1e-5)
    # The 3x3 presampler sits centred in its 7x15 channel; the padding stays zero.
    assert (X[:, 0, 2:5, 6:9] > 0).all() and X[:, 0].sum() == X[:, 0, 2:5, 6:9].sum()
    assert preprocessor.model_kwargs(X) == {"in_channels": 7, "cell_height": 7, "cell_width": 15}


def test_fused_vector(per_ring_schema):
    preprocessor = PreprocessFused()
    df = load_dataframe(per_ring_schema, preprocessor.required_columns())
    X = preprocessor.fit_transform(df)

    assert X.shape == (df.height, 100 + 7 * 7 * 15) and X.dtype == np.float32
    np.testing.assert_allclose(X.sum(axis=1), 1.0, rtol=1e-5)
    assert preprocessor.model_kwargs(X) == {"n_rings": 100, "cell_shape": (7, 7, 15)}


def test_every_registered_model_declares_a_preprocessor():
    assert {"MLP", "CNN2D", "Fused"} <= set(available_models())
    for name in available_models():
        model_cls = get_model(name)
        assert hasattr(model_cls, "preprocessor"), name
        assert model_cls.preprocessor().required_columns()
