import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.pipelines.ingestion import run_ingestion_pipeline
from repositories.upload_repository import UploadRepository

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(self, poll_interval: int = 60):
        self.interval = poll_interval
        self._shutdown = False
        self._task: asyncio.Task | None = None
        self._processing = False

    async def start(self) -> None:
        self._shutdown = False
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Ingestion worker started (poll interval: %ds)", self.interval)

    async def stop(self) -> None:
        logger.info("Stopping ingestion worker...")
        self._shutdown = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Ingestion worker stopped")

    async def _process_loop(self) -> None:
        settings = get_settings()
        upload_dir = Path(settings.temp_upload_dir)
        repo = UploadRepository()

        while not self._shutdown:
            try:
                if self._processing:
                    await asyncio.sleep(self.interval)
                    continue

                job = repo.get_next_queued()
                if job is None:
                    await asyncio.sleep(self.interval)
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
        pdf_path = (upload_dir / f"{job_id}.pdf").resolve()

        logger.info("Worker picked up job %s: %s", job_id, original_filename)

        try:
            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"PDF file not found on disk at {pdf_path}. "
                    "The server may have restarted and lost the temporary file. Please re-upload the document."
                )

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
