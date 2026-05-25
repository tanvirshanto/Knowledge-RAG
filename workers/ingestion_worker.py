import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.pipelines.ingestion import run_ingestion_pipeline
from repositories.upload_repository import UploadRepository
from repositories.base import get_supabase

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(self, poll_interval: int = 60):
        self.interval = poll_interval
        self._shutdown = False
        self._task: asyncio.Task | None = None
        self._processing = False

    async def start(self) -> None:
        logger.info("IngestionWorker starting...")
        if self._task is not None:
            logger.info("IngestionWorker already running")
            return
        self._shutdown = False
        self._task = asyncio.create_task(self._process_loop())
        logger.info("IngestionWorker started with poll interval %s seconds", self.interval)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._shutdown = True
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("IngestionWorker stopped")

    async def _process_loop(self) -> None:
        repo = UploadRepository()
        settings = get_settings()
        upload_dir = Path(settings.temp_upload_dir)

        while not self._shutdown:
            try:
                await asyncio.sleep(self.interval)
                if self._processing:
                    continue

                job = repo.get_next_queued()
                logger.info("Worker picked up job %s", job)
                if not job:
                    continue

                self._processing = True
                try:
                    await self._process_job(job, repo, upload_dir)
                finally:
                    self._processing = False
            except Exception as exc:
                logger.exception("Error in worker process loop")
                await asyncio.sleep(self.interval)

    async def _process_job(
        self,
        job: dict,
        repo: UploadRepository,
        upload_dir: Path,
    ) -> None:
        job_id = job["id"]
        original_filename = job.get("original_filename", "document.pdf")
        storage_path = job.get("storage_path") or f"uploads/{job_id}.pdf"
        pdf_path = (upload_dir / f"{job_id}.pdf").resolve()

        logger.info("Worker picked up job %s: %s (storage_path: %s)", job_id, original_filename, storage_path)

        try:
            logger.info("Downloading PDF from Supabase storage path: %s", storage_path)
            try:
                content_bytes = get_supabase().storage.from_("mediRag").download(storage_path)
            except Exception as e:
                raise FileNotFoundError(
                    f"PDF file for job {job_id} could not be downloaded from storage path '{storage_path}': {e}"
                )

            # Ensure upload_dir exists
            upload_dir.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(content_bytes)
            logger.info("Saved PDF locally to: %s", pdf_path)

            await asyncio.to_thread(
                run_ingestion_pipeline,
                job_id=job_id,
                pdf_path=pdf_path,
                filename=original_filename,
            )
            current = repo.get_by_id(job_id)
            if current and current.get("status") != "COMPLETED":
                repo.update_status(
                    job_id,
                    "COMPLETED",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
            logger.info("Job %s completed successfully", job_id)

        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            repo.update_status(
                job_id,
                "FAILED",
                error_message=str(exc)[:500],
            )
        finally:
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                    logger.info("Cleaned up temporary PDF file: %s", pdf_path)
                except Exception as e:
                    logger.warning("Failed to delete temporary PDF file %s: %s", pdf_path, e)
