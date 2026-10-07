"""`python -m pose_app.worker`: process analysis jobs until SIGTERM / Ctrl-C."""

import logging
import signal
import threading

from dotenv import find_dotenv, load_dotenv

from .config import LogConfig, WorkerConfig, configure_logging
from .jobs import HttpJobClient
from .worker import Worker


def main() -> int:
    # .env in the current folder or a parent; real environment variables win.
    load_dotenv(find_dotenv(usecwd=True))
    configure_logging(LogConfig.from_env())
    workerConfig = WorkerConfig.from_env()
    stop = threading.Event()

    stop_on_signals(stop)

    jobs = HttpJobClient(workerConfig.api_url, workerConfig.api_token, workerConfig.api_timeout_s, workerConfig.api_retry_s)
    try:
        Worker(workerConfig, jobs).run(stop)
    finally:
        jobs.close()
    return 0

def stop_on_signals(stop: threading.Event) -> None:
    def request_stop(signum, _frame):
        logging.getLogger(__name__).info("%s: stopping after the current job",
                                            signal.Signals(signum).name)
        stop.set()
        
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    

raise SystemExit(main())
