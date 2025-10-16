from __future__ import annotations
import logging
import sys
from .config import APP_TITLE, LOG_LEVEL
from .domain import AppState
from .services.yolo_service import YoloService
from .controllers.controllers import NavController, LabelController, YoloController
from .services.device_manager import DeviceManager
from .view import View

def setup_logging(level: int = LOG_LEVEL) -> None:
    fmt = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def build_container() -> View:
    state = AppState()
    device_manager = DeviceManager()
    yolo_service = YoloService(state)
    nav_ctrl = NavController(state)
    label_ctrl = LabelController(state)
    yolo_ctrl = YoloController(state, yolo_service, device_manager, label_ctrl)
    view = View(state, nav_ctrl, label_ctrl, yolo_ctrl)
    def _on_close():
        try:
            yolo_service.shutdown()
        except Exception:
            pass
        view.on_quit()
    view.protocol("WM_DELETE_WINDOW", _on_close)
    return view

def main(argv=None):
    setup_logging()
    logging.getLogger(__name__).info("Starting %s", APP_TITLE)
    view = build_container()
    try:
        view.mainloop()
    finally:
        try:
            view.yolo.service.shutdown()
        except Exception:
            pass
        logging.getLogger(__name__).info("Exiting %s", APP_TITLE)

if __name__ == "__main__":
    sys.exit(main())
