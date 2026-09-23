"""Command-line interface: parse options, pick a camera, start the app."""

import argparse
import sys
from typing import List, Optional, Sequence, Tuple

from .camera_selection import choose_camera
from .cameras import Camera, discover_cameras
from .config import Settings
from .ui.display import enable_hidpi
from .ui.view import VIEW_MODES


def _size_arg(text: str) -> Tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got '{text}'")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam pose estimation with joint-angle readout.")

    cam = p.add_argument_group("camera")
    cam.add_argument("--device", type=int, default=None,
                     help="camera index to use (skips the interactive prompt)")
    cam.add_argument("--list-devices", action="store_true",
                     help="list the detected cameras and exit")
    cam.add_argument("--max-index", type=int, default=8,
                     help="highest camera index to probe (default: 8)")
    cam.add_argument("--capture-width", type=int, default=1280)
    cam.add_argument("--capture-height", type=int, default=720)
    cam.add_argument("--no-mirror", action="store_true", help="do not mirror the camera image")

    model = p.add_argument_group("model")
    model.add_argument("--model", default="yolo11n-pose.pt",
                       help="Ultralytics pose weights (n/s/m/l/x, default: yolo11n-pose.pt)")
    model.add_argument("--conf", type=float, default=0.5,
                       help="person detection confidence threshold (default: 0.5)")
    model.add_argument("--kpt-conf", type=float, default=0.5,
                       help="minimum keypoint confidence for an angle (default: 0.5)")
    model.add_argument("--imgsz", type=int, default=640, help="inference size (default: 640)")
    model.add_argument("--infer-device", default=None,
                       help="torch device for inference, e.g. cpu, 0, mps")
    model.add_argument("--smooth", type=float, default=0.4,
                       help="angle smoothing factor, 1.0 disables it (default: 0.4)")

    disp = p.add_argument_group("display")
    disp.add_argument("--fullscreen", action="store_true", help="start in full screen")
    disp.add_argument("--window-size", type=_size_arg, default=None,
                      help="initial window size, e.g. 1280x720 (default: fit the screen)")
    disp.add_argument("--view", choices=VIEW_MODES, default="auto",
                      help="initial view mode (default: auto)")

    rec = p.add_argument_group("recording")
    rec.add_argument("--record", default=None, help="write the view to this .mp4 file")
    rec.add_argument("--record-size", type=_size_arg, default=(1280, 720),
                     help="fixed frame size of the recording (default: 1280x720)")
    return p.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        model=args.model, conf=args.conf, kpt_conf=args.kpt_conf, imgsz=args.imgsz,
        infer_device=args.infer_device,
        capture_size=(args.capture_width, args.capture_height),
        mirror=not args.no_mirror,
        fullscreen=args.fullscreen, window_size=args.window_size, view=args.view,
        smooth=args.smooth,
        record_path=args.record, record_size=args.record_size,
    )


def _resolve_camera(args: argparse.Namespace, cameras: List[Camera]) -> Optional[Camera]:
    if args.device is None:
        return choose_camera(cameras)
    camera = next((c for c in cameras if c.index == args.device), None)
    if camera is None:
        print(f"Camera index {args.device} is not available. Detected: "
              f"{[c.index for c in cameras]}", file=sys.stderr)
    return camera


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    enable_hidpi()

    print("Scanning for cameras ...")
    cameras = discover_cameras(max_index=args.max_index)
    if not cameras:
        print("No working webcam was found. Check that the device is connected "
              "and not in use by another application.", file=sys.stderr)
        return 1

    if args.list_devices:
        for cam in cameras:
            print(cam.describe())
        return 0

    camera = _resolve_camera(args, cameras)
    if camera is None:
        return 1
    print(f"\nUsing {camera.describe()}")

    from .app import PoseApp   # deferred: only load the model stack when needed
    try:
        app = PoseApp(settings_from_args(args), camera)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    return app.run()
