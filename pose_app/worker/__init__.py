"""Queue worker: runs analyses requested by the web app.

    analysis_job row (MySQL) --claim--> run_analysis() on the files in its job_dir
        -> outputs written next to them -> row SUCCEEDED / FAILED

Start with `python -m pose_app.worker`; settings are in config.py and the
table contract in schema.sql.
"""
