-- Job queue shared by the web app and the analysis worker (MySQL 8.0+).
--
-- The web app owns this table (put it in its migrations, e.g. Flyway) and may
-- add its own columns such as user_id. It inserts a QUEUED row per upload and
-- reads status, progress and result; it may set status = 'CANCELLED' on a
-- QUEUED or RUNNING job. The worker claims QUEUED rows and writes the rest.
--
-- Files live on a disk both sides can reach. Paths here are relative to the
-- media root, which each side configures for itself (the worker: WORKER_MEDIA_ROOT),
-- so the two may mount it in different places. The worker reads the inputs in
-- job_dir and writes its outputs next to them: summary.json, annotated.mp4 and
-- pose_cache.npz (inputs must not use these names, nor reps.csv / report.html).

CREATE TABLE analysis_job (
    id              BIGINT        NOT NULL AUTO_INCREMENT PRIMARY KEY,
    status          VARCHAR(16)   NOT NULL DEFAULT 'QUEUED',    -- QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED

    -- Written by the web app
    station         VARCHAR(32)   NOT NULL,            -- e.g. 'skierg'
    job_dir         VARCHAR(1024) NOT NULL,            -- e.g. 'analysis/{username}/{uuid}'
    video_file      VARCHAR(255)  NOT NULL,             -- file name in job_dir
    telemetry_file  VARCHAR(255)  NULL,                 -- machine data CSV in job_dir, if any
    options         JSON          NULL,                -- see JOB_OPTIONS in pose_app/worker/jobs.py
    created_at      DATETIME(3)   NOT NULL DEFAULT CURRENT_TIMESTAMP(3),

    -- Written by the worker
    attempts        INT           NOT NULL DEFAULT 0,
    worker_id       VARCHAR(128)  NULL,
    progress_stage  VARCHAR(16)   NULL,                 -- pose, video
    progress_pct    TINYINT       NULL,                 -- within the stage; NULL if unknown
    heartbeat_at    DATETIME(3)   NULL,
    started_at      DATETIME(3)   NULL,
    finished_at     DATETIME(3)   NULL,
    result          JSON          NULL,                 -- on SUCCEEDED, see Worker._result
    error_kind      VARCHAR(16)   NULL,                 -- 'input': the upload can't be analysed
                                                        -- 'internal': our fault, retried
    error_code      VARCHAR(32)   NULL,                 -- 'input' only: see ERROR_CODES in pose_app/analysis/api.py
    error_message   TEXT          NULL,                 -- details in English, for logs

    KEY idx_analysis_job_queue (status, id)
);
