"""Machine telemetry (erg power, pace, stroke rate) fused with the video analysis.

    load_telemetry(path) -> Telemetry        read an app export or a PM5 log (raw)
    pm5_telemetry(session) -> Telemetry       the same from an already decoded PM5 log
    clean(tel) -> Telemetry                   gaps, held values, dropouts
    align(tel, starts, durations, ...)        video clock -> machine clock
    attach / analyze_machine                  per-stroke m_* metrics, splits, checks
"""

from .clean import clean
from .fusion import (MACHINE_KEYS, MachineAnalysis, analyze_machine, attach,
                     machine_drift_rules)
from .model import Telemetry
from .readers import load_telemetry, pm5_telemetry
from .sync import Alignment, align

__all__ = ["Telemetry", "load_telemetry", "pm5_telemetry", "clean", "Alignment", "align",
           "attach", "analyze_machine", "MachineAnalysis", "machine_drift_rules", "MACHINE_KEYS"]
