import dataclasses

import numpy as np
import pytest

from neuralnet_lightning.binning import ETA_EDGES, ET_EDGES, Region
from neuralnet_lightning.data.loader import load_dataframe
from neuralnet_lightning.data.schema import ET, ETA, LABEL, ROW_ID


def test_projection_keeps_only_requested_columns(per_ring_schema, listed_schema):
    assert load_dataframe(per_ring_schema, ["ring_3", "cl_rhad"]).columns == ["ring_3", "cl_rhad", ET, ETA, LABEL]
    assert load_dataframe(listed_schema, ["ring_3"]).columns == ["ring_3", ET, ETA, ROW_ID, LABEL]


def test_region_filter(per_ring_schema):
    region = Region(1, 0)
    df = load_dataframe(per_ring_schema, ["ring_0"], region)
    everything = load_dataframe(per_ring_schema, ["ring_0"])

    et, abs_eta = everything[ET].to_numpy(), np.abs(everything[ETA].to_numpy())
    expected = ((et >= ET_EDGES[1]) & (et < ET_EDGES[2]) & (abs_eta >= ETA_EDGES[0]) & (abs_eta < ETA_EDGES[1])).sum()
    assert 0 < df.height == expected
    assert df[ET].min() >= ET_EDGES[1] and df[ET].max() < ET_EDGES[2]
    assert np.abs(df[ETA].to_numpy()).max() < ETA_EDGES[1]


def test_empty_region_raises(per_ring_schema):
    with pytest.raises(ValueError, match="No rows"):
        load_dataframe(per_ring_schema, ["ring_0"], Region(4, 0))  # synthetic Et stops at 45 GeV


def test_unknown_column_raises(per_ring_schema):
    with pytest.raises(KeyError, match="not in the dataset"):
        load_dataframe(per_ring_schema, ["cl_does_not_exist"])


def test_cell_images_missing_from_a_dataset_fail_early(listed_schema):
    """CNN2D/Fused on a dataset without cell images fails in the loader, not mid-training."""
    from neuralnet_lightning.preprocess.cnn2d import PreprocessCNN2D

    with pytest.raises(KeyError, match=r"cl_cells_presampler.*not in the dataset"):
        load_dataframe(listed_schema, PreprocessCNN2D().required_columns())


def test_max_files_caps_each_folder(per_ring_schema):
    capped = dataclasses.replace(per_ring_schema, max_files=1)
    assert len(capped.files()) == 1
    assert load_dataframe(capped, ["ring_0"])[LABEL].unique().to_list() == [0]  # JF17 sorts first


def test_region_validation():
    with pytest.raises(ValueError, match="outside the grid"):
        Region(5, 0)
