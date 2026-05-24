import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_upload_dir(upload_dir: str) -> Path:
    path = Path(upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_upload(content: bytes, job_id: str, upload_dir: str) -> Path:
    dest = Path(upload_dir) / f"{job_id}.pdf"
    dest.write_bytes(content)
    logger.info("Saved upload %s to %s", job_id, dest)
    return dest


def delete_upload(file_path: Path) -> None:
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info("Deleted upload file %s", file_path)
    except OSError as exc:
        logger.warning("Could not delete upload file %s: %s", file_path, exc)
