import importlib.util
import inspect
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check nano-crucible classes
from crucible.resources.datasets import DatasetOperations
from crucible.resources.postprocessing import PostprocessingOperations
from crucible.resources.samples import SampleOperations
from crucible.models import Dataset
import utils


def _load_consumer_module():
    module_path = Path(__file__).resolve().parent.parent / "src" / "consumer-rga-analysis.py"
    spec = importlib.util.spec_from_file_location("consumer_rga_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


consumer = _load_consumer_module()


# ---------------------------------------------------------------------------
# 1. Nano-Crucible 4.0.1 Signature Compatibility Checks
# ---------------------------------------------------------------------------
def test_crucible_v4_signature_compatibility():
    """
    Verify that our calls match exact nano-crucible 4.0.1 method signatures
    and that none of the target methods are deprecated/removed stubs.
    """
    # DatasetOperations.download
    sig_dl = inspect.signature(DatasetOperations.download)
    assert not getattr(DatasetOperations.download, "__removed__", False)
    bound_dl = sig_dl.bind(
        None,
        "0tdataset000000000000000001",
        include=["*.zip", "*.ZIP"],
        no_record=True,
        output_dir="./",
    )
    assert bound_dl.arguments["include"] == ["*.zip", "*.ZIP"]
    assert bound_dl.arguments["no_record"] is True

    # DatasetOperations.create
    sig_create = inspect.signature(DatasetOperations.create)
    assert not getattr(DatasetOperations.create, "__removed__", False)
    mock_ds = MagicMock(spec=Dataset)
    bound_create = sig_create.bind(
        None,
        mock_ds,
        files=["/tmp/a.txt"],
        scientific_metadata={},
        wait_for_ingestion_response=False,
        skip_ingestion=True,
    )
    assert bound_create.arguments["files"] == ["/tmp/a.txt"]
    assert bound_create.arguments["skip_ingestion"] is True

    # DatasetOperations.add_file
    sig_add_file = inspect.signature(DatasetOperations.add_file)
    assert not getattr(DatasetOperations.add_file, "__removed__", False)
    bound_add_file = sig_add_file.bind(
        None,
        "0tdataset000000000000000001",
        "/tmp/file.txt",
        wait_for_ingestion_response=False,
        skip_ingestion=True,
        replace=True,
    )
    assert bound_add_file.arguments["replace"] is True
    assert bound_add_file.arguments["skip_ingestion"] is True

    # DatasetOperations.link
    sig_link = inspect.signature(DatasetOperations.link)
    assert not getattr(DatasetOperations.link, "__removed__", False)
    bound_link = sig_link.bind(
        None,
        "0tparent000000000000000001",
        "0tchild0000000000000000001",
    )
    assert len(bound_link.arguments) >= 3  # self, parent, child

    # DatasetOperations.link_sample
    sig_link_sample = inspect.signature(DatasetOperations.link_sample)
    assert not getattr(DatasetOperations.link_sample, "__removed__", False)
    bound_link_s = sig_link_sample.bind(
        None,
        "0tdataset000000000000000001",
        "0tsample0000000000000000001",
    )
    assert len(bound_link_s.arguments) >= 3

    # DatasetOperations.add_thumbnail
    sig_thumb = inspect.signature(DatasetOperations.add_thumbnail)
    assert not getattr(DatasetOperations.add_thumbnail, "__removed__", False)
    bound_thumb = sig_thumb.bind(
        None,
        "0tdataset000000000000000001",
        "/tmp/thumb.png",
        thumbnail_name="thumb.png",
    )
    assert bound_thumb.arguments["thumbnail_name"] == "thumb.png"

    # SampleOperations.list
    sig_samples_list = inspect.signature(SampleOperations.list)
    assert not getattr(SampleOperations.list, "__removed__", False)
    bound_s_list = sig_samples_list.bind(
        None,
        dataset_mfid="0tdataset000000000000000001",
    )
    assert bound_s_list.arguments["dataset_mfid"] == "0tdataset000000000000000001"

    # SampleOperations.update_scientific_metadata
    sig_s_update_md = inspect.signature(SampleOperations.update_scientific_metadata)
    assert not getattr(SampleOperations.update_scientific_metadata, "__removed__", False)
    bound_s_md = sig_s_update_md.bind(
        None,
        "0tsample0000000000000000001",
        metadata={"outgas_area": 1e-6},
    )
    assert bound_s_md.arguments["metadata"] == {"outgas_area": 1e-6}

    # PostprocessingOperations.update
    sig_pp_update = inspect.signature(PostprocessingOperations.update)
    assert not getattr(PostprocessingOperations.update, "__removed__", False)
    bound_pp = sig_pp_update.bind(
        None,
        "run-12345",
        "completed",
        error=None,
    )
    assert bound_pp.arguments["status"] == "completed"


# ---------------------------------------------------------------------------
# 2. utils.get_raw_data Unit Tests
# ---------------------------------------------------------------------------
def test_get_raw_data_nested_subfolder():
    """
    Test downloading and extracting a zip with a nested folder named after stem.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_mfid = "0trawdataset0000000000000001"
        zip_dir = os.path.join(tmpdir, raw_mfid)
        os.makedirs(zip_dir, exist_ok=True)
        zip_path = os.path.join(zip_dir, "batch_data.zip")

        # Create zip with internal structure batch_data/sample_TEY.txt
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("batch_data/sample_TEY_DarkPD_0.txt", "time\tval\n1\t2\n")
            zf.writestr("batch_data/sample_RGA_histogram_0.txt", "time\tmz\n1\t2\n")

        mock_client = MagicMock()
        mock_client.datasets.download.return_value = [zip_path]

        data_zip, extracted_path = utils.get_raw_data(mock_client, raw_mfid, work_root=tmpdir)

        assert data_zip == zip_path
        assert os.path.isdir(extracted_path)
        assert os.path.basename(extracted_path) == "batch_data"
        assert os.path.exists(os.path.join(extracted_path, "sample_TEY_DarkPD_0.txt"))

        mock_client.datasets.download.assert_called_once_with(
            raw_mfid,
            include=["*.zip", "*.ZIP"],
            no_record=True,
            output_dir=tmpdir,
        )


def test_get_raw_data_single_subfolder_descend():
    """
    Test extraction where zip contains a single folder with arbitrary name and no root txt files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_mfid = "0trawdataset0000000000000002"
        zip_dir = os.path.join(tmpdir, raw_mfid)
        os.makedirs(zip_dir, exist_ok=True)
        zip_path = os.path.join(zip_dir, "archive.zip")

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("inner_folder/data.txt", "content\n")

        mock_client = MagicMock()
        mock_client.datasets.download.return_value = [zip_path]

        data_zip, extracted_path = utils.get_raw_data(mock_client, raw_mfid, work_root=tmpdir)
        assert os.path.basename(extracted_path) == "inner_folder"
        assert os.path.exists(os.path.join(extracted_path, "data.txt"))


def test_get_raw_data_no_zip_raises():
    """
    Test when no zip files are returned by download - raises FileNotFoundError.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_client = MagicMock()
        mock_client.datasets.download.return_value = ["file1.txt", "file2.pdf"]

        with pytest.raises(FileNotFoundError, match="No ingested .zip file found to download"):
            utils.get_raw_data(mock_client, "0trawdataset0000000000000003", work_root=tmpdir)


def test_get_raw_data_exception_raised():
    """
    Test that download exceptions are re-raised and not swallowed.
    """
    mock_client = MagicMock()
    mock_client.datasets.download.side_effect = RuntimeError("GCS network down")

    with pytest.raises(RuntimeError, match="GCS network down"):
        utils.get_raw_data(mock_client, "0trawdataset0000000000000004")


# ---------------------------------------------------------------------------
# 3. fetch_existing_child_map Unit Tests
# ---------------------------------------------------------------------------
def test_fetch_existing_child_map():
    mock_client = MagicMock()
    parent_mfid = "0tparent000000000000000001"

    child1 = {"unique_id": "0tchild000000000000000001"}
    child2 = {"unique_id": "0tchild000000000000000002"}
    child3_unlinked = {"unique_id": "0tchild000000000000000003"}

    mock_client.datasets.list_children.return_value = [child1, child2, child3_unlinked]

    def mock_sample_list(dataset_mfid=None, **kwargs):
        assert dataset_mfid is not None
        assert "dataset_id" not in kwargs  # Confirm dataset_id is NOT used
        if dataset_mfid == child1["unique_id"]:
            return [{"sample_name": "SAMPLE_A", "unique_id": "0tsample000000000000000001"}]
        elif dataset_mfid == child2["unique_id"]:
            return [{"sample_name": "SAMPLE_B", "unique_id": "0tsample000000000000000002"}]
        elif dataset_mfid == child3_unlinked["unique_id"]:
            return []  # Unlinked child
        return []

    mock_client.samples.list.side_effect = mock_sample_list

    child_map = consumer.fetch_existing_child_map(mock_client, parent_mfid)

    mock_client.datasets.list_children.assert_called_once_with(parent_mfid, limit=None)
    assert child_map == {
        "SAMPLE_A": "0tchild000000000000000001",
        "SAMPLE_B": "0tchild000000000000000002",
    }
    assert "0tchild000000000000000003" not in child_map.values()


# ---------------------------------------------------------------------------
# 4. create_sample_dataset: New Child Path
# ---------------------------------------------------------------------------
def test_create_sample_dataset_new_child():
    mock_client = MagicMock()
    sample_entry = {
        "sample_name": "TF003580",
        "sample_id": "0t000000000000000000003580",
    }
    spot = "Spot1"
    ds = {
        "unique_id": "0tparent000000000000000001",
        "dataset_name": "RAW_RUN_01",
        "project": {"project_id": "proj-extracted-99"},
        "project_id": None,
        "owner": {"unique_id": "owner-extracted-42"},
        "owner_orcid": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        rga_raw_file = os.path.join(tmpdir, "TF003580_RGA_histogram_001.txt")
        with open(rga_raw_file, "w") as f:
            f.write("mock rga content")

        mock_measurement = MagicMock()
        mock_measurement.files = [rga_raw_file, os.path.join(tmpdir, "TF003580_derived.txt")]
        mock_measurement.metadata = {
            "rga_source": os.path.basename(rga_raw_file),
            "param": 123,
        }
        mock_measurement.outgas_area = 2.45e-6
        mock_thumb = os.path.join(tmpdir, "TF003580_multipanel.png")
        with open(mock_thumb, "wb") as f:
            f.write(b"PNG")
        mock_measurement.thumbnail = mock_thumb

        with patch("rga_batch.process_sample", return_value=mock_measurement):
            res = consumer.create_sample_dataset(
                sample_entry=sample_entry,
                spot=spot,
                ds=ds,
                directory=tmpdir,
                crucible_client=mock_client,
                sample_sub_dataset_id_map={},
                measured_samples={"TF003580"},
            )

        assert res is not None
        sample_name, sds_mfid = res
        assert sample_name == "TF003580"
        assert len(sds_mfid) == 26

        # Check datasets.create call
        mock_client.datasets.create.assert_called_once()
        call_args, call_kwargs = mock_client.datasets.create.call_args
        created_ds = call_args[0]
        assert isinstance(created_ds, Dataset)
        assert created_ds.instrument_id == "als-bl12012"
        assert created_ds.instrument_mfid == "0tby4fs3xxr97000f4bc4t8rhc"
        assert getattr(created_ds, "instrument_name", None) is None
        assert created_ds.project_id == "proj-extracted-99"
        assert created_ds.owner == "owner-extracted-42"
        assert call_kwargs["files"] == mock_measurement.files
        assert call_kwargs["skip_ingestion"] is True

        # Check parent-child link
        mock_client.datasets.link.assert_called_once_with(ds["unique_id"], created_ds.unique_id)

        # Check sample link
        mock_client.datasets.link_sample.assert_called_once_with(
            created_ds.unique_id, sample_entry["sample_id"]
        )

        # Check outgas area update
        mock_client.samples.update_scientific_metadata.assert_called_once_with(
            sample_entry["sample_id"],
            metadata={
                "outgas_area": 2.45e-6,
                "outgas_area_analysis_reference": f"dataset: {created_ds.unique_id}",
            },
        )

        # Check thumbnail
        mock_client.datasets.add_thumbnail.assert_called_once_with(
            sds_mfid,
            mock_thumb,
            thumbnail_name="TF003580_multipanel.png",
        )


def test_create_sample_dataset_no_valid_sample_id():
    """
    Test when sample_id is None or not a valid MFID - skips sample linking gracefully.
    """
    mock_client = MagicMock()
    sample_entry = {
        "sample_name": "TF003580",
        "sample_id": "not-an-mfid",
    }
    spot = "Spot1"
    ds = {
        "unique_id": "0tparent000000000000000001",
        "dataset_name": "RAW_RUN_01",
        "project_id": "proj-1",
        "owner_orcid": "owner-1",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        rga_raw = os.path.join(tmpdir, "raw.txt")
        with open(rga_raw, "w") as f:
            f.write("test")

        mock_m = MagicMock()
        mock_m.files = [rga_raw]
        mock_m.metadata = {"rga_source": "raw.txt"}
        mock_m.outgas_area = 1.0e-5
        mock_m.thumbnail = None

        with patch("rga_batch.process_sample", return_value=mock_m):
            res = consumer.create_sample_dataset(
                sample_entry=sample_entry,
                spot=spot,
                ds=ds,
                directory=tmpdir,
                crucible_client=mock_client,
                sample_sub_dataset_id_map={},
                measured_samples={"TF003580"},
            )

        assert res is not None
        mock_client.datasets.create.assert_called_once()
        mock_client.datasets.link.assert_called_once()
        # link_sample and samples.update_scientific_metadata must NOT be called
        mock_client.datasets.link_sample.assert_not_called()
        mock_client.samples.update_scientific_metadata.assert_not_called()


# ---------------------------------------------------------------------------
# 5. create_sample_dataset: Existing Child Rerun Path
# ---------------------------------------------------------------------------
def test_create_sample_dataset_existing_child():
    mock_client = MagicMock()
    existing_mfid = "0texistingchild000000000001"
    sample_entry = {
        "sample_name": "TF003580",
        "sample_id": "0t000000000000000000003580",
    }
    spot = "Spot1"
    ds = {
        "unique_id": "0tparent000000000000000001",
        "dataset_name": "RAW_RUN_01",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "f1.txt")
        file2 = os.path.join(tmpdir, "f2.txt")
        with open(file1, "w") as f:
            f.write("1")
        with open(file2, "w") as f:
            f.write("2")

        mock_m = MagicMock()
        mock_m.files = [file1, file2]
        mock_m.metadata = {"rga_source": "f1.txt"}
        mock_m.outgas_area = 3.14e-6
        mock_m.thumbnail = None

        with patch("rga_batch.process_sample", return_value=mock_m):
            res = consumer.create_sample_dataset(
                sample_entry=sample_entry,
                spot=spot,
                ds=ds,
                directory=tmpdir,
                crucible_client=mock_client,
                sample_sub_dataset_id_map={"TF003580": existing_mfid},
                measured_samples={"TF003580"},
            )

        assert res == ("TF003580", existing_mfid)
        mock_client.datasets.create.assert_not_called()
        mock_client.datasets.update.assert_called_once()
        mock_client.datasets.update_scientific_metadata.assert_called_once()

        # Files refreshed via add_file with replace=True, skip_ingestion=True
        assert mock_client.datasets.add_file.call_count == 2
        mock_client.datasets.add_file.assert_any_call(
            existing_mfid,
            file1,
            wait_for_ingestion_response=False,
            skip_ingestion=True,
            replace=True,
        )
        mock_client.datasets.add_file.assert_any_call(
            existing_mfid,
            file2,
            wait_for_ingestion_response=False,
            skip_ingestion=True,
            replace=True,
        )


# ---------------------------------------------------------------------------
# 6. run_rga_analysis End-to-End Success & Failure Tests
# ---------------------------------------------------------------------------
def test_run_rga_analysis_success():
    with tempfile.TemporaryDirectory() as work_root:
        raw_mfid = "0trawdataset0000000000000005"
        run_id = "run-test-success-001"
        body = json.dumps({"dataset_mfid": raw_mfid, "run_id": run_id}).encode("utf-8")

        mock_ch = MagicMock()
        mock_method = MagicMock()
        mock_method.delivery_tag = 42
        mock_conn = MagicMock()
        # Immediately invoke threadsafe callbacks
        mock_conn.add_callback_threadsafe.side_effect = lambda cb: cb()

        mock_client = MagicMock()
        mock_client.datasets.get.return_value = {
            "unique_id": raw_mfid,
            "dataset_name": "BATCH_01",
            "project": {"project_id": "proj-1"},
            "owner": {"unique_id": "owner-1"},
            "scientific_metadata": {
                "samples": {
                    "Spot1": {
                        "sample_name": "S1",
                        "sample_id": "0tsample000000000000000001",
                    }
                }
            },
        }

        # Create dummy run_dir inside work_root to test cleanup
        raw_run_dir = os.path.join(work_root, raw_mfid)
        os.makedirs(raw_run_dir, exist_ok=True)
        dummy_zip = os.path.join(raw_run_dir, "batch.zip")
        with open(dummy_zip, "w") as f:
            f.write("zip content")

        with patch.object(consumer, "get_raw_data", return_value=(dummy_zip, raw_run_dir)), \
             patch.object(consumer, "validate_sample_files", return_value={"S1"}), \
             patch.object(consumer, "fetch_existing_child_map", return_value={}), \
             patch.object(consumer, "create_sample_dataset", return_value=("S1", "0t000000000000000000000011")):

            consumer.run_rga_analysis(
                ch=mock_ch,
                method=mock_method,
                body=body,
                connection=mock_conn,
                crucible_client=mock_client,
                work_root=work_root,
            )

        # Check Status updates: "running" then "completed"
        assert mock_client.postprocessing.update.call_count == 2
        mock_client.postprocessing.update.assert_any_call(run_id, "running", error=None)
        mock_client.postprocessing.update.assert_any_call(run_id, "completed", error=None)

        # Basic ack called
        mock_ch.basic_ack.assert_called_once_with(delivery_tag=42)

        # Temporary mkdtemp directory inside work_root cleaned up (while pre-existing unrelated files stay untouched)
        assert os.listdir(work_root) == [raw_mfid]
        assert os.path.exists(dummy_zip)


def test_run_rga_analysis_failure():
    with tempfile.TemporaryDirectory() as work_root:
        raw_mfid = "0trawdataset0000000000000006"
        run_id = "run-test-failure-002"
        body = json.dumps({"dataset_mfid": raw_mfid, "run_id": run_id}).encode("utf-8")

        mock_ch = MagicMock()
        mock_method = MagicMock()
        mock_method.delivery_tag = 99
        mock_conn = MagicMock()
        mock_conn.add_callback_threadsafe.side_effect = lambda cb: cb()

        mock_client = MagicMock()
        mock_client.datasets.get.side_effect = RuntimeError("Crucible DB connection failed")

        raw_run_dir = os.path.join(work_root, raw_mfid)
        os.makedirs(raw_run_dir, exist_ok=True)

        consumer.run_rga_analysis(
            ch=mock_ch,
            method=mock_method,
            body=body,
            connection=mock_conn,
            crucible_client=mock_client,
            work_root=work_root,
        )

        # Status updates: "running" then "failed" with error
        assert mock_client.postprocessing.update.call_count == 2
        mock_client.postprocessing.update.assert_any_call(run_id, "running", error=None)
        mock_client.postprocessing.update.assert_any_call(
            run_id, "failed", error="Crucible DB connection failed"
        )

        # Publish to failed queue and basic ack
        mock_ch.basic_publish.assert_called_once_with(
            exchange="",
            routing_key="rga-analysis-failed",
            body=json.dumps({"dataset_mfid": raw_mfid, "run_id": run_id}),
        )
        mock_ch.basic_ack.assert_called_once_with(delivery_tag=99)

        # Temporary mkdtemp directory cleaned up without deleting unrelated siblings
        assert os.listdir(work_root) == [raw_mfid]


def test_run_rga_analysis_malformed_json_acks_and_dead_letters():
    with tempfile.TemporaryDirectory() as work_root:
        mock_ch = MagicMock()
        mock_method = MagicMock()
        mock_method.delivery_tag = 77
        mock_conn = MagicMock()
        mock_conn.add_callback_threadsafe.side_effect = lambda cb: cb()
        mock_client = MagicMock()

        consumer.run_rga_analysis(
            ch=mock_ch,
            method=mock_method,
            body=b"{not-valid-json",
            connection=mock_conn,
            crucible_client=mock_client,
            work_root=work_root,
        )

        mock_ch.basic_publish.assert_called_once_with(
            exchange="",
            routing_key="rga-analysis-failed",
            body=b"{not-valid-json",
        )
        mock_ch.basic_ack.assert_called_once_with(delivery_tag=77)
        assert os.listdir(work_root) == []


def test_extract_helpers_normalize_empty_strings_to_none():
    ds = {"project": None, "project_id": "  ", "owner": None, "owner_orcid": ""}
    assert consumer._extract_project_id(ds) is None
    assert consumer._extract_owner_ref(ds) is None
