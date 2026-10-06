"""Queue worker: runs analyses requested by the web app.

    POST /internal/jobs/claim --> run_analysis() on the uploads the job names
        -> outputs written next to them, report sent back -> complete / fail

Start with `python -m pose_app.worker`; settings are in config.py and the
API contract in jobs.py. The worker talks to the web app only through its
internal HTTP API, and shares only the media folder with it.
"""
