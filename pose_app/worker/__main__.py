"""`python -m pose_app.worker`: process analysis jobs until SIGTERM / Ctrl-C."""

import logging
import signal
import threading

from dotenv import find_dotenv, load_dotenv

from .config import WorkerConfig
from .jobs import JobStore
from .worker import Worker


def main() -> int:
    logging.basicConfig(level=logging.INFO, filename="worker.log",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # .env in the current folder or a parent; real environment variables win.
    load_dotenv(find_dotenv(usecwd=True))
    cfg = WorkerConfig.from_env()
    stop = threading.Event()

    def request_stop(signum, _frame):
        logging.getLogger(__name__).info("%s: stopping after the current job",
                                         signal.Signals(signum).name)
        stop.set()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    jobs = JobStore(cfg.database_url)
    try:
        Worker(cfg, jobs).run(stop)
    finally:
        jobs.close()
    return 0


raise SystemExit(main())
