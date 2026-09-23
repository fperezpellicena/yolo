"""PoseApp: ties camera, estimator and UI together and handles the keyboard."""

import dataclasses
import sys
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from .cameras import Camera, open_camera
from .config import Settings
from .estimator import PoseEstimator
from .fps import FpsMeter
from .recorder import Recorder
from .skeleton import empty_angles
from .smoothing import AngleSmoother
from .ui.annotations import draw_joint_angles
from .ui.display import Display
from .ui.help_overlay import draw_help
from .ui.toast import Toast
from .ui.view import VIEW_MODES, ViewState, render_view

WINDOW_NAME = "Pose estimation"
ESC = "\x1b"


class PoseApp:
    def __init__(self, settings: Settings, camera: Camera):
        self.settings = settings
        self.estimator = PoseEstimator(settings.model, settings.conf, settings.imgsz,
                                       settings.infer_device, settings.kpt_conf)
        self.cap = open_camera(camera, settings.capture_size)
        self.display = Display(WINDOW_NAME, settings.window_size, settings.fullscreen)
        self.recorder = Recorder(settings.record_path, settings.record_size) \
            if settings.record_path else None
        self.smoother = AngleSmoother(settings.smooth)
        self.fps = FpsMeter()
        self.toast = Toast()

        self.view_index = VIEW_MODES.index(settings.view)
        self.mirror = settings.mirror
        self.paused = False
        self.show_help = False
        self.snapshots = 0
        self.running = True

        self._state: Optional[ViewState] = None     # last processed frame
        self._canvas: Optional[np.ndarray] = None   # last rendered view (no overlays)

        self._actions: Dict[str, Callable[[], None]] = {
            "q": self.quit,
            ESC: self.escape,
            "f": self.toggle_fullscreen,
            "v": self.next_view,
            "m": self.toggle_mirror,
            " ": self.toggle_pause,
            "h": self.toggle_help,
            "s": self.save_snapshot,
        }

    # ------------------------------------------------------------------ loop

    def run(self) -> int:
        self.toast.show("Press h for keyboard shortcuts", 4.0)
        try:
            while self.running:
                new_frame = False
                if not self.paused:
                    if not self._process_next_frame():
                        break
                    new_frame = True

                if self._state is None:
                    self._handle_key(cv2.waitKey(10))
                    continue

                # Re-render every iteration (even when paused) so resizing and
                # full-screen changes take effect immediately.
                self._canvas = self._render()
                if new_frame and self.recorder is not None:
                    self.recorder.write(self._canvas)
                self.display.show(self._with_overlays(self._canvas))

                self._handle_key(cv2.waitKey(30 if self.paused else 1))
                if not self.display.is_open():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
        return 0

    def _process_next_frame(self) -> bool:
        ok, raw = self.cap.read()
        if not ok or raw is None:
            print("Camera stopped delivering frames.", file=sys.stderr)
            return False
        frame = cv2.flip(raw, 1) if self.mirror else raw

        people, annotated = self.estimator.estimate(frame)
        for person in people:
            draw_joint_angles(annotated, person, self.settings.kpt_conf)

        if people:
            angles = self.smoother(people[0].angles)
        else:
            self.smoother.reset()
            angles = empty_angles()

        self.fps.tick()
        self._state = ViewState(frame=frame, annotated=annotated,
                                n_people=len(people), angles=angles)
        return True

    def _render(self) -> np.ndarray:
        state = dataclasses.replace(self._state, fps=self.fps.value, paused=self.paused)
        width, height = self.display.canvas_size()
        return render_view(width, height, state, VIEW_MODES[self.view_index])

    def _with_overlays(self, canvas: np.ndarray) -> np.ndarray:
        """Help and toasts go on a copy, so recordings/snapshots stay clean."""
        if not (self.show_help or self.toast.active):
            return canvas
        shown = canvas.copy()
        if self.show_help:
            draw_help(shown)
        self.toast.draw(shown)
        return shown

    def _handle_key(self, key: int) -> None:
        key &= 0xFF
        char = chr(key).lower() if key < 128 else ""
        action = self._actions.get(char)
        if action is not None:
            action()

    def close(self) -> None:
        self.cap.release()
        if self.recorder is not None:
            self.recorder.close()
        self.display.close()

    # --------------------------------------------------------------- actions

    def quit(self) -> None:
        self.running = False

    def escape(self) -> None:
        if self.display.fullscreen:
            self.display.toggle_fullscreen()
            self.toast.show("Windowed")
        else:
            self.quit()

    def toggle_fullscreen(self) -> None:
        self.display.toggle_fullscreen()
        self.toast.show("Full screen  (Esc to leave)" if self.display.fullscreen
                        else "Windowed")

    def next_view(self) -> None:
        self.view_index = (self.view_index + 1) % len(VIEW_MODES)
        self.toast.show(f"View: {VIEW_MODES[self.view_index]}")

    def toggle_mirror(self) -> None:
        self.mirror = not self.mirror
        self.smoother.reset()
        self.toast.show("Mirror on" if self.mirror else "Mirror off")

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.fps.restart()
        self.toast.show("Paused" if self.paused else "Resumed")

    def toggle_help(self) -> None:
        self.show_help = not self.show_help

    def save_snapshot(self) -> None:
        if self._canvas is None:
            return
        self.snapshots += 1
        path = f"pose_snapshot_{self.snapshots:03d}.png"
        cv2.imwrite(path, self._canvas)
        self.toast.show(f"Saved {path}")
        print(f"Saved {path}")
