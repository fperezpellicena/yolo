"""The OpenCV window: creation, full screen, and querying its drawable size."""

import platform
from typing import Optional, Tuple

import cv2

Size = Tuple[int, int]


def enable_hidpi() -> None:
    """On Windows, opt out of bitmap scaling so the UI is sharp on HiDPI screens.
    Call before any window is created."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def screen_size() -> Optional[Size]:
    system = platform.system()
    size: Optional[Size] = None
    try:
        if system == "Windows":
            import ctypes
            user32 = ctypes.windll.user32
            size = (int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1)))
        elif system != "Darwin":          # Tk and OpenCV's Cocoa loop don't mix
            import tkinter
            root = tkinter.Tk()
            root.withdraw()
            size = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
            root.destroy()
    except Exception:
        return None
    # A multi-monitor desktop can report the combined width - ignore that.
    if size and size[0] > 0 and size[1] > 0 and size[0] / size[1] <= 2.4:
        return size
    return None


class Display:
    """OpenCV window whose drawable size is queried every frame."""

    MIN_SIZE = 64

    def __init__(self, name: str, size: Optional[Size] = None, fullscreen: bool = False):
        self.name = name
        self.screen = screen_size()
        if size is None:
            size = self._default_size()
        self.windowed_size = size
        self._last = size
        self._seen_visible = False
        self.fullscreen = False

        flags = cv2.WINDOW_NORMAL
        flags |= getattr(cv2, "WINDOW_FREERATIO", 0)      # let the image fill the window
        flags |= getattr(cv2, "WINDOW_GUI_NORMAL", 0)     # no Qt toolbar / status bar
        cv2.namedWindow(name, flags)
        cv2.resizeWindow(name, *size)
        if fullscreen:
            self.toggle_fullscreen()

    def _default_size(self) -> Size:
        if self.screen:
            sw, sh = self.screen
            w = min(1600, int(sw * 0.85))
            return w, min(int(sh * 0.8), int(w * 9 / 16))
        return 1280, 720

    def toggle_fullscreen(self) -> None:
        if not self.fullscreen:
            self.windowed_size = self._last
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.fullscreen = True
        else:
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.name, *self.windowed_size)
            self.fullscreen = False

    def canvas_size(self) -> Size:
        """Current drawable area of the window, with sensible fallbacks."""
        w = h = 0
        try:
            _, _, w, h = cv2.getWindowImageRect(self.name)
        except (cv2.error, AttributeError):
            pass
        if w >= self.MIN_SIZE and h >= self.MIN_SIZE:
            self._last = (int(w), int(h))
            return self._last
        if self.fullscreen and self.screen:
            return self.screen
        return self._last

    def show(self, image) -> None:
        cv2.imshow(self.name, image)

    def is_open(self) -> bool:
        try:
            visible = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return False
        if visible >= 1:
            self._seen_visible = True
            return True
        # Some backends never report visibility; only treat the window as
        # closed once we have seen it open.
        return not self._seen_visible

    def close(self) -> None:
        cv2.destroyAllWindows()
