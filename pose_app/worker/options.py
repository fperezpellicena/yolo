"""A job's options, as the web app sends them, turned into AnalysisOptions."""

from ..analysis.api import AnalysisError, AnalysisOptions
from ..analysis.tracking import SELECT_MODES
from .config import WorkerConfig
from .jobs import Job

# Keys the web app may put in a job's options, all optional. Model and
# hardware settings are the worker's own (see config.py), not per job.
JOB_OPTIONS = {
    "start": "clip start in seconds",
    "end": "clip end in seconds",
    "rotate": "0, 90, 180 or 270",
    "athlete": "'largest' or 'center'",
    "athlete_point": "[x, y] in pixels: follow the person there in the first frame",
    "telemetry_offset": "machine seconds at video second 0, if automatic sync fails",
    "thresholds": "{rule_id: value} overriding the station's defaults",
    "video": "false to skip the annotated video",
}

# Converts each JSON value in a job's options to what AnalysisOptions expects.
_PARSE = {
    "start": float,
    "end": float,
    "rotate": lambda v: _one_of(int(v), (0, 90, 180, 270)),
    "athlete": lambda v: _one_of(v, SELECT_MODES),
    "athlete_point": lambda v: (tuple(float(x) for x in v)
                                if isinstance(v, list) and len(v) == 2 else _bad(v)),
    "telemetry_offset": float,
    "thresholds": lambda v: {str(k): float(x) for k, x in v.items()},
    "video": lambda v: v if isinstance(v, bool) else _bad(v),
}
assert set(_PARSE) == set(JOB_OPTIONS)


def build_analysis_options(job: Job, cfg: WorkerConfig) -> AnalysisOptions:
    unknown = set(job.options) - set(JOB_OPTIONS)
    if unknown:
        raise AnalysisError("invalid_job", f"Unknown job options: {sorted(unknown)}.")
    parsed = {}
    for key, value in job.options.items():
        if value is None:
            continue
        try:
            parsed[key] = _PARSE[key](value)
        except (TypeError, ValueError, AttributeError):
            raise AnalysisError("invalid_job", f"Invalid job option {key}={value!r}: "
                                               f"expected {JOB_OPTIONS[key]}.") from None
    # No report.html or reps.csv: the web app builds its report from the summary.
    return AnalysisOptions(station=job.station, model=cfg.model, imgsz=cfg.imgsz,
                           device=cfg.device, report=False, csv=False, **parsed)


def _one_of(value, allowed):
    return value if value in allowed else _bad(value)


def _bad(value):
    raise ValueError(value)
