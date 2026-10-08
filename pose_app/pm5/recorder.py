"""Record a PM5 over Bluetooth into a raw log (needs `pip install bleak`).

The PM5 accepts one app connection at a time: close ErgData (or any other
rowing app) first, and wake the monitor. Every notification of the rowing
service is written as it arrives, so the log keeps what this version cannot
decode yet (e.g. characteristics added by newer firmware).
"""

import asyncio
import logging
import time
from typing import Callable, Dict, Optional

from .log import RawLogWriter
from .protocol import (DEVICE_INFO, ROWING_SERVICE, STROKE_DATA, WORKOUT_SUMMARY, StrokeData,
                       parse, uuid)

log = logging.getLogger(__name__)

SCAN_S = 8.0                 # one discovery pass
SCAN_PASSES = 6
IDLE_STOP_S = 600.0          # stop this long after the last stroke
AFTER_END_S = 10.0           # ...and this long after the monitor's end-of-workout summary


def is_pm5(device, adv) -> bool:
    """macOS often leaves the device name empty and only advertises it, so look at both,
    and at the Concept2 UUIDs in the advertisement."""
    names = [n for n in (getattr(device, "name", None), getattr(adv, "local_name", None)) if n]
    uuids = [u.lower() for u in (getattr(adv, "service_uuids", None) or [])]
    return any(n.startswith("PM5") for n in names) or any(u.startswith("ce06") for u in uuids)


async def find_pm5(address: Optional[str] = None):
    from bleak import BleakScanner
    for attempt in range(1, SCAN_PASSES + 1):
        seen = await BleakScanner.discover(timeout=SCAN_S, return_adv=True)
        hits = [(d, a) for d, a in seen.values()
                if (d.address == address if address else is_pm5(d, a))]
        if hits:
            device, adv = max(hits, key=lambda x: x[1].rssi or -999)
            log.info("found %s (rssi %s)", device.name or adv.local_name, adv.rssi)
            return device
        log.info("scan %d: no PM5 yet (wake it, close other rowing apps)", attempt)
    return None


async def read_device_info(client) -> Dict[str, str]:
    info = {}
    for short, key in DEVICE_INFO.items():
        try:
            info[key] = (await client.read_gatt_char(uuid(short))).decode(errors="replace").strip("\x00 ")
        except Exception:                       # not every firmware exposes every field
            pass
    return info


async def record(path: str, address: Optional[str] = None, minutes: Optional[float] = None,
                 stop: Optional[asyncio.Event] = None,
                 on_stroke: Optional[Callable[[StrokeData], None]] = None) -> int:
    """Record until `stop` is set, the piece ends, strokes stop for IDLE_STOP_S or
    `minutes` pass. Returns the number of notifications written."""
    from bleak import BleakClient
    stop = stop or asyncio.Event()
    device = await find_pm5(address)
    if device is None:
        raise RuntimeError("No PM5 found. Wake it (press a button) and close ErgData or any "
                           "other app connected to it.")
    state = {"last_stroke": time.monotonic(), "ended": None}
    with RawLogWriter(path) as writer:
        def handler(short: int):
            def on_notify(_char, data: bytearray):
                writer.notification(short, bytes(data))
                if short == STROKE_DATA:
                    state["last_stroke"] = time.monotonic()
                    stroke = parse(short, bytes(data))
                    if on_stroke and stroke is not None:
                        on_stroke(stroke)
                elif short == WORKOUT_SUMMARY and state["ended"] is None:
                    state["ended"] = time.monotonic()
            return on_notify

        async with BleakClient(device) as client:
            info = await read_device_info(client)
            writer.meta(device={"name": device.name, "address": device.address, **info},
                        recorder="pose_app.pm5")
            service = client.services.get_service(uuid(ROWING_SERVICE))
            if service is None:
                raise RuntimeError(f"{device.name} has no Concept2 rowing service; is it a PM5?")
            subscribed = []
            for ch in service.characteristics:
                if "notify" in ch.properties:
                    short = int(ch.uuid[4:8], 16)
                    await client.start_notify(ch, handler(short))
                    subscribed.append(f"{short:04x}")
            log.info("recording %s: %s", device.name, " ".join(subscribed))
            deadline = time.monotonic() + minutes * 60 if minutes else None
            while client.is_connected and not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                now = time.monotonic()
                if state["ended"] is not None and now - state["ended"] > AFTER_END_S:
                    log.info("piece ended on the monitor")
                    break
                if now - state["last_stroke"] > IDLE_STOP_S:
                    log.info("no strokes for %.0f minutes; stopping", IDLE_STOP_S / 60)
                    break
                if deadline and now > deadline:
                    break
        return writer.count
