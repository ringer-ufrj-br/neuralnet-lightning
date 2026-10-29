import numpy as np
import polars as pl
import pytest

from neuralnet_lightning.data.labels import validate_files
from neuralnet_lightning.data.loader import load_dataframe
from neuralnet_lightning.data.schema import ET, ETA, LABEL, ROW_ID, ring_name


def test_per_ring_columns_are_canonical(per_ring_schema):
    frame = per_ring_schema.scan(per_ring_schema.files())
    names = per_ring_schema.canonical_columns(frame.collect_schema().names())

    assert [ring_name(i) for i in range(100)] == names[:100]
    assert {ET, ETA, LABEL, "cl_rhad"} <= set(names)
    # Source names reachable through an alias, and the scan's bookkeeping, are not offered.
    assert not {"cl_ring_0", "cl_et", "cl_eta", "file_path", ROW_ID} & set(names)


def test_listed_rings_are_canonical(listed_schema):
    frame = listed_schema.scan(listed_schema.files())
    names = listed_schema.canonical_columns(frame.collect_schema().names())

    assert ring_name(99) in names and ROW_ID in names
    assert not {"Cluster.ringsE", "target", "id"} & set(names)


def test_label_from_file_name(per_ring_schema):
    df = load_dataframe(per_ring_schema, ["ring_0"])
    # Files are read in sorted order: JF17 (background) before Zee (signal).
    assert df[LABEL].to_list() == [0] * 300 + [1] * 300


def test_label_from_column(listed_schema):
    df = load_dataframe(listed_schema, ["ring_0"])
    assert df[LABEL].to_list() == [1] * 200 + [0] * 200
    assert df[ROW_ID].to_list() == list(range(400))


def test_both_layouts_give_the_same_rings(per_ring_schema, listed_schema):
    """The same events stored either way must come out identical."""
    df = load_dataframe(per_ring_schema, [ring_name(i) for i in range(100)])
    listed_path = per_ring_schema.data_path + "_listed.parquet"
    pl.DataFrame({
        "ringsE": df.select([ring_name(i) for i in range(100)]).to_numpy().tolist(),
        "et": df[ET], "eta": df[ETA], "target": df[LABEL],
    }).write_parquet(listed_path)

    listed = load_dataframe(type(listed_schema)(data_path=listed_path, rings_col="ringsE",
                                                et_col="et", eta_col="eta", label_col="target"),
                            [ring_name(i) for i in range(100)])
    cols = [ring_name(i) for i in range(100)]
    np.testing.assert_allclose(listed.select(cols).to_numpy(), df.select(cols).to_numpy())


def test_unlabelled_file_is_rejected():
    with pytest.raises(ValueError, match="Could not determine label"):
        validate_files(["data/mystery_sample.parquet"])
