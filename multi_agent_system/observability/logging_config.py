# multi-agent-system/observability/logging_config.py
import logging
import sys

def configure_logging(level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.handlers = [handler]
