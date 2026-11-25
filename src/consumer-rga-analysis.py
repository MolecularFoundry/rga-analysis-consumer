import os
import shutil
import json
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from crucible import CrucibleClient
from crucible.models import Dataset
from dotenv import load_dotenv
import mfid

try:
    from crucible.utils.identifiers import is_mfid
except ImportError:
    def is_mfid(val):
        return isinstance(val, str) and len(val) == 26

import rga_batch
from utils import setup_pika_client, get_raw_data, get_secret, sanitize_metadata

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Constants
DEFAULT_CRUCIBLE_API_URL = "https://crucible.lbl.gov/api/v4"
ALS_BL12012_INSTRUMENT_ID = "als-bl12012"
ALS_BL12012_INSTRUMENT_MFID = "0tby4fs3xxr97000f4bc4t8rhc"

# Module-level client / connection handles (populated in main())
client = None
connection = None
channel = None


class ThreadLocalCrucibleClient:
    """
    Thread-local wrapper around CrucibleClient so concurrent threads in
    fetch_existing_child_map and run_rga_analysis do not share state/sessions.
    """
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
        self._local = threading.local()

    @property
    def _client(self):
        if not hasattr(self._local, "client"):
            self._local.client = CrucibleClient(api_url=self.api_url, api_key=self.api_key)
        return self._local.client

    def __getattr__(self, name):
        return getattr(self._client, name)

    def __repr__(self):
        return f"<ThreadLocalCrucibleClient api_url={self.api_url!r}>"


def _extract_project_id(ds):
    proj = ds.get("project")
    if isinstance(proj, dict) and proj.get("project_id"):
        return proj["project_id"]
    val = ds.get("project_id")
    return val.strip() if isinstance(val, str) and val.strip() else None


def _extract_owner_ref(ds):
    owner = ds.get("owner")
    if isinstance(owner, dict):
        val = owner.get("unique_id") or owner.get("orcid") or ds.get("owner_orcid")
        return val.strip() if isinstance(val, str) and val.strip() else None
    if isinstance(owner, str) and owner.strip():
        return owner.strip()
    val = ds.get("owner_orcid")
    return val.strip() if isinstance(val, str) and val.strip() else None


def _extract_scientific_metadata(ds):
    raw = ds.get("scientific_metadata") or {}
    if isinstance(raw, dict) and isinstance(raw.get("scientific_metadata"), dict):
        return raw["scientific_metadata"]
    return raw


def _report_run_status(crucible_client, run_id, status, error=None):
    if run_id is None or crucible_client is None:
        return
    try:
        err_msg = error[:4000] if error else None
        crucible_client.postprocessing.update(run_id, status, error=err_msg)
    except Exception as status_err:
        logger.warning(
            f"Failed to report postprocessing status '{status}' for run {run_id}: {status_err}"
        )


def fetch_existing_child_map(crucible_client, parent_id):
    def fetch_sample(sds):
        # A child whose sample link never got made is not usable as a rerun target,
        # but it must not take the whole batch down with it.
        samples = crucible_client.samples.list(dataset_mfid=sds["unique_id"])
        if not samples:
            logger.warning(f"Existing child {sds['unique_id']} has no linked sample, ignoring")
            return None
        return samples[0]["sample_name"], sds["unique_id"]

    # limit=None: a truncated page would hide existing children and send them
    # back down the create path, which 409s.
    children = crucible_client.datasets.list_children(parent_id, limit=None)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(fetch_sample, children)
    return dict(r for r in results if r is not None)


def is_empty_spot(sample_name):
    return sample_name is None or str(sample_name).strip().lower() in ("", "nan", "none")


def validate_sample_files(sample_dictionary, directory):
    """
    Check every sample on the holder has exactly one TEY and one RGA file before
    anything is uploaded, so a bad batch fails with no partial children created.

    Returns the set of sample names that were measured in this batch.
    """
    measured = set()
    invalid = []

    for spot, entry in sample_dictionary.items():
        sample_name = entry["sample_name"]
        if is_empty_spot(sample_name):
            continue

        tey_files, rga_files = rga_batch.match_sample_files(sample_name, directory)

        if not tey_files and not rga_files:
            logger.info(f"[{spot} {sample_name}] not measured in this batch, skipping")
        elif len(tey_files) == 1 and len(rga_files) == 1:
            measured.add(sample_name)
        else:
            invalid.append((spot, sample_name, tey_files, rga_files))

    if invalid:
        lines = ["Expected exactly 1 TEY and 1 RGA file per sample, found:"]
        for spot, sample_name, tey_files, rga_files in invalid:
            lines.append(f"  [{spot}] {sample_name}: {len(tey_files)} TEY, {len(rga_files)} RGA")
            for f in tey_files + rga_files:
                lines.append(f"    {os.path.basename(f)}")
        raise ValueError("\n".join(lines))

    return measured


def create_sample_dataset(
    sample_entry,
    spot,
    ds,
    directory,
    crucible_client,
    sample_sub_dataset_id_map,
    measured_samples,
):
    raw_sample_id = sample_entry.get("sample_id")
    sample_id = raw_sample_id if isinstance(raw_sample_id, str) and is_mfid(raw_sample_id) else None
    sample_name = sample_entry["sample_name"]

    # Skip empty/placeholder spots so we don't create datasets with no sample name.
    if is_empty_spot(sample_name):
        logger.warning(f"[{spot}] No sample name for this spot, skipping dataset creation")
        return None

    # Skip spots with no data files so we don't create empty datasets.
    if sample_name not in measured_samples:
        logger.warning(f"[{spot} {sample_name}] No files found for sample, skipping dataset creation")
        return None

    m = rga_batch.process_sample(sample_name, directory=directory, overwrite=True)
    sample_files = m.files

    existing_mfid = sample_sub_dataset_id_map.get(sample_name)
    sds_mfid = existing_mfid or mfid.mfid()[0]

    # Timestamp off the raw RGA file: the derived files were written moments ago and
    # would record the processing time rather than the measurement time.
    rga_raw = os.path.join(directory, m.metadata["rga_source"])
    if os.path.exists(rga_raw):
        timestamp = datetime.fromtimestamp(os.path.getmtime(rga_raw), tz=timezone.utc).isoformat()
    else:
        timestamp = datetime.now(timezone.utc).isoformat()

    if existing_mfid:
        # POSTing a known id returns 409, so refresh in place.
        crucible_client.datasets.update(sds_mfid, timestamp=timestamp)
        crucible_client.datasets.update_scientific_metadata(sds_mfid, sanitize_metadata(m.metadata))
        for file_path in sample_files:
            crucible_client.datasets.add_file(
                sds_mfid,
                file_path,
                wait_for_ingestion_response=False,
                skip_ingestion=True,
                replace=True,
            )
        logger.info(f"  [{spot}] {sample_name} already exists, updated {sds_mfid}")
    else:
        sds = Dataset(
            unique_id=sds_mfid,
            dataset_name=f"RGATEY_{ds['dataset_name']}_{spot}_{sample_name}",
            instrument_id=ALS_BL12012_INSTRUMENT_ID,
            instrument_mfid=ALS_BL12012_INSTRUMENT_MFID,
            measurement="automated_RGA_TEY_run",
            project_id=_extract_project_id(ds),
            data_type="automated_RGA_TEY_run",
            owner=_extract_owner_ref(ds),
            timestamp=timestamp,
        )

        crucible_client.datasets.create(
            sds,
            files=sample_files,
            scientific_metadata=sanitize_metadata(m.metadata),
            wait_for_ingestion_response=False,
            skip_ingestion=True,
        )

        crucible_client.datasets.link(ds["unique_id"], sds.unique_id)
        if sample_id:
            crucible_client.datasets.link_sample(sds.unique_id, sample_id)
        else:
            logger.warning(f"  [{spot}] {sample_name} has no valid sample MFID; skipping dataset-sample link")
        logger.info(f"  [{spot}] {sample_name} -> {sds_mfid}")

    # Record the corrected outgassing area on the sample's scientific metadata
    outgas_area = m.outgas_area
    if outgas_area is not None and sample_id:
        md = {
            "outgas_area": float(outgas_area),
            "outgas_area_analysis_reference": f"dataset: {sds_mfid}",
        }
        crucible_client.samples.update_scientific_metadata(sample_id, metadata=md)
        logger.info(f"  [{spot}] {sample_name} outgas_area={outgas_area:.3e} -> sample {sample_id}")
    elif outgas_area is not None and not sample_id:
        logger.warning(
            f"  [{spot}] {sample_name} outgas_area={outgas_area:.3e} calculated, but no valid sample_id to update metadata"
        )
    else:
        logger.warning(f"  [{spot}] {sample_name} no outgas_area found, skipping metadata update")

    if m.thumbnail:
        tn_name = os.path.basename(m.thumbnail)
        try:
            with open(m.thumbnail, "rb"):
                pass
            replaced_tn = False
            if existing_mfid:
                try:
                    existing_tns = crucible_client.datasets.get_thumbnails(sds_mfid) or []
                    match_tn = next(
                        (t for t in existing_tns if isinstance(t, dict) and t.get("thumbnail_name") == tn_name and t.get("id") is not None),
                        None,
                    )
                    if match_tn is not None:
                        crucible_client.datasets.update_thumbnail(
                            sds_mfid, match_tn["id"], image=m.thumbnail, thumbnail_name=tn_name
                        )
                        replaced_tn = True
                except Exception as tn_err:
                    logger.warning(f"  [warn] could not check/update existing thumbnail on {sds_mfid}: {tn_err}")
            if not replaced_tn:
                crucible_client.datasets.add_thumbnail(sds_mfid, m.thumbnail, thumbnail_name=tn_name)
        except FileNotFoundError:
            logger.warning(f"  [warn] thumbnail not found, skipping: {m.thumbnail}")

    return sample_name, sds_mfid


def run_rga_analysis(ch, method, body, connection, crucible_client=None, work_root="./"):
    if crucible_client is None:
        crucible_client = client

    message = None
    raw_mfid = None
    run_id = None
    run_work_dir = None

    try:
        raw_text = body.decode("utf-8").strip() if isinstance(body, (bytes, bytearray)) else str(body).strip()
        message = json.loads(raw_text)
        if not isinstance(message, dict):
            raise ValueError(f"Expected JSON object message, got {type(message).__name__}")
        raw_mfid = message.get("dataset_mfid") or message.get("dsid")
        if not raw_mfid:
            raise ValueError("Message missing 'dataset_mfid' / 'dsid'")
        run_id = message.get("run_id")

        logger.info(f"Received message for dataset {raw_mfid} (run_id={run_id}), starting processing")
        _report_run_status(crucible_client, run_id, "running")

        os.makedirs(work_root, exist_ok=True)
        run_work_dir = tempfile.mkdtemp(prefix=f"rga_{raw_mfid}_", dir=work_root)

        # get the dataset SQL record
        og_dataset = crucible_client.datasets.get(raw_mfid, include_metadata=True)

        # get the raw data files into isolated per-invocation directory
        _, directory = get_raw_data(crucible_client, raw_mfid, work_root=run_work_dir)

        scientific_metadata = _extract_scientific_metadata(og_dataset)
        sample_dictionary = scientific_metadata.get("samples") or {}
        sample_positions = list(sample_dictionary.keys())

        # Validate before anything is uploaded, so a bad batch creates no children.
        measured_samples = validate_sample_files(sample_dictionary, directory)
        logger.info(f"{len(measured_samples)} samples measured in this batch")

        # upload to crucible - following Ed's workflow
        logger.info("Fetching existing child map...")
        sample_sub_dataset_id_map = fetch_existing_child_map(crucible_client, raw_mfid)

        logger.info("Analyzing and creating sample datasets...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(
                executor.map(
                    lambda sample_position: create_sample_dataset(
                        sample_dictionary[sample_position],
                        sample_position,
                        og_dataset,
                        directory,
                        crucible_client,
                        sample_sub_dataset_id_map,
                        measured_samples,
                    ),
                    sample_positions,
                )
            )

        logger.info("Crucible upload complete.")
        _report_run_status(crucible_client, run_id, "completed")

        # acknowledge the message
        def on_success():
            if ch is not None and hasattr(ch, "basic_ack") and method is not None:
                ch.basic_ack(delivery_tag=method.delivery_tag)

        if connection is not None and hasattr(connection, "add_callback_threadsafe"):
            connection.add_callback_threadsafe(on_success)
        else:
            on_success()

    except Exception as err:
        logger.error(f"Error processing dataset {raw_mfid}: {err}")
        _report_run_status(crucible_client, run_id, "failed", error=str(err))
        failed_payload = json.dumps(message) if isinstance(message, dict) else body

        def on_failure():
            if ch is not None and hasattr(ch, "basic_publish"):
                ch.basic_publish(
                    exchange="",
                    routing_key="rga-analysis-failed",
                    body=failed_payload,
                )
            if ch is not None and hasattr(ch, "basic_ack") and method is not None:
                ch.basic_ack(delivery_tag=method.delivery_tag)

        if connection is not None and hasattr(connection, "add_callback_threadsafe"):
            connection.add_callback_threadsafe(on_failure)
        else:
            on_failure()

    finally:
        if run_work_dir and os.path.isdir(run_work_dir):
            try:
                shutil.rmtree(run_work_dir)
                logger.info(f"Removed temporary run directory: {run_work_dir}")
            except OSError as e:
                logger.warning(f"Could not remove temporary run directory {run_work_dir}: {e}")


def callback(ch, method, properties, body):
    """
    Expects a RMQ message with:
    dsid / dataset_mfid: The dataset ID that the processing request was made for
    run_id:              Optional postprocessing run ID
    """
    thread = threading.Thread(
        target=run_rga_analysis,
        args=(ch, method, body, connection, client),
    )
    thread.start()


def main():
    global client, connection, channel

    load_dotenv()
    rmq_host = os.environ.get("RMQ_HOST", "localhost")
    rmq_port = int(os.environ.get("RMQ_PORT", 5672))
    rmq_pw = get_secret("RABBITMQ_DEFAULT_PW", "rabbitmq_default_pw/versions/1")

    crucible_api_url = os.environ.get("CRUCIBLE_API_URL", DEFAULT_CRUCIBLE_API_URL)
    crucible_api_key = (
        os.environ.get("CRUCIBLE_API_KEY")
        or os.environ.get("ADMIN_APIKEY")
        or get_secret("CRUCIBLE_API_KEY", "crucible_apikey_sa_ingestion/versions/latest")
    )

    num_cores = os.cpu_count()
    logger.info(f"Starting consumer with {num_cores=}")

    # Initialize thread-safe Crucible client
    client = ThreadLocalCrucibleClient(api_url=crucible_api_url, api_key=crucible_api_key)

    # Initialize RabbitMQ connection
    connection, channel = setup_pika_client(rmq_host, rmq_port, rmq_pw)
    queues_needed = ["rga-analysis", "rga-analysis-failed"]
    for q in queues_needed:
        channel.queue_declare(queue=q)

    # Subscribe to queue (prefetch_count=1 prevents concurrent batch OOM and disk races)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue="rga-analysis",
        auto_ack=False,
        on_message_callback=callback,
    )

    logger.info(" [*] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()


if __name__ == "__main__":
    main()
