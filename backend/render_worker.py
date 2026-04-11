from __future__ import annotations

import logging
import threading
import time

from backend.job_queue import JobQueueConfig, create_job_queue
from backend.logging_utils import configure_logging


logger = logging.getLogger(__name__)


def run_render_worker(
    *,
    config: JobQueueConfig | None = None,
    stop_event: threading.Event | None = None,
    sleep_s: float = 1.0,
) -> int:
    """Run the job renderer outside the API process."""
    configure_logging(role="render")
    engine, _sessionmaker, worker = create_job_queue(config or JobQueueConfig())
    worker.start()
    logger.info("Render worker started.")

    try:
        while stop_event is None or not stop_event.is_set():
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        logger.info("Render worker interrupted.")
    finally:
        logger.info("Stopping render worker.")
        worker.stop()
        engine.dispose()

    return 0


def main() -> int:
    return run_render_worker()


if __name__ == "__main__":
    raise SystemExit(main())
