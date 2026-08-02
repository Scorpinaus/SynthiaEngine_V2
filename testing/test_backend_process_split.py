import threading
from dataclasses import replace

import backend.main as main
import backend.jobs.render_worker as render_worker


class _FakeEngine:
    def __init__(self):
        self.disposed = False

    def dispose(self):
        self.disposed = True


class _FakeWorker:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def _clear_job_state(application):
    for name in ("job_engine", "job_sessionmaker", "job_worker", "job_worker_started"):
        if hasattr(application.state, name):
            delattr(application.state, name)


def test_api_startup_can_disable_embedded_worker(monkeypatch):
    settings = replace(
        main.app.state.settings,
        api=replace(main.app.state.settings.api, start_embedded_worker=False),
    )
    application = main.create_app(settings)
    _clear_job_state(application)
    fake_engine = _FakeEngine()
    fake_worker = _FakeWorker()
    fake_sessionmaker = object()

    def _fake_create_job_queue(config):
        return fake_engine, fake_sessionmaker, fake_worker

    monkeypatch.setattr(main, "create_job_queue", _fake_create_job_queue)

    try:
        main._startup_job_queue(application)

        assert application.state.job_engine is fake_engine
        assert application.state.job_sessionmaker is fake_sessionmaker
        assert application.state.job_worker is fake_worker
        assert application.state.job_worker_started is False
        assert fake_worker.started is False
    finally:
        main._shutdown_job_queue(application)
        _clear_job_state(application)


def test_api_startup_starts_embedded_worker_by_default(monkeypatch):
    settings = replace(
        main.app.state.settings,
        api=replace(main.app.state.settings.api, start_embedded_worker=True),
    )
    application = main.create_app(settings)
    _clear_job_state(application)
    fake_engine = _FakeEngine()
    fake_worker = _FakeWorker()

    def _fake_create_job_queue(config):
        return fake_engine, object(), fake_worker

    monkeypatch.setattr(main, "create_job_queue", _fake_create_job_queue)

    try:
        main._startup_job_queue(application)

        assert application.state.job_worker_started is True
        assert fake_worker.started is True
    finally:
        main._shutdown_job_queue(application)
        _clear_job_state(application)


def test_render_worker_entrypoint_starts_and_stops_worker(monkeypatch):
    fake_engine = _FakeEngine()
    fake_worker = _FakeWorker()

    def _fake_create_job_queue(config):
        return fake_engine, object(), fake_worker

    stop_event = threading.Event()
    stop_event.set()
    monkeypatch.setattr(render_worker, "create_job_queue", _fake_create_job_queue)

    result = render_worker.run_render_worker(stop_event=stop_event, sleep_s=0)

    assert result == 0
    assert fake_worker.started is True
    assert fake_worker.stopped is True
    assert fake_engine.disposed is True
