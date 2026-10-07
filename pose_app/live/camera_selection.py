"""Interactive console prompt for picking a camera."""

from typing import List

from .cameras import Camera


def choose_camera(cameras: List[Camera]) -> Camera:
    """Print the discovered cameras and ask the user to pick one."""
    print("\nAvailable video input devices:")
    for position, cam in enumerate(cameras, start=1):
        print(f"  {position}. {cam.describe()}")

    if len(cameras) == 1:
        print("\nOnly one device found - using it.")
        return cameras[0]

    while True:
        try:
            raw = input(f"\nSelect a device [1-{len(cameras)}] (default 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo selection made - using the first device.")
            return cameras[0]
        if not raw:
            return cameras[0]
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(cameras):
                return cameras[choice - 1]
            # Also accept the raw device index (e.g. "2" for /dev/video2).
            for cam in cameras:
                if cam.index == choice:
                    return cam
        print("Invalid choice, try again.")
