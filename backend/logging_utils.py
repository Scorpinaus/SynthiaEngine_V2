import logging
import os


_LOG_FORMAT = "%(asctime)s %(levelname)s [%(synthia_role)s] %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"
_LOG_RECORD_FACTORY_INSTALLED = False
_LOG_ROLE = os.getenv("SYNTHA_LOG_ROLE", "app")


def _install_log_record_factory() -> None:
    global _LOG_RECORD_FACTORY_INSTALLED
    if _LOG_RECORD_FACTORY_INSTALLED:
        return

    previous_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        record.synthia_role = _LOG_ROLE
        return record

    logging.setLogRecordFactory(record_factory)
    _LOG_RECORD_FACTORY_INSTALLED = True


def configure_logging(level: int = logging.INFO, *, role: str | None = None) -> None:
    global _LOG_ROLE
    if role:
        _LOG_ROLE = role
    elif os.getenv("SYNTHA_LOG_ROLE"):
        _LOG_ROLE = os.environ["SYNTHA_LOG_ROLE"]

    _install_log_record_factory()

    root_logger = logging.getLogger()
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
    if root_logger.handlers:
        for handler in root_logger.handlers:
            if handler.formatter is None:
                handler.setFormatter(formatter)
        if root_logger.level > level:
            root_logger.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
