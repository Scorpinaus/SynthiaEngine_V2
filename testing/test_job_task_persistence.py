from backend.jobs.queue import (
    JobQueueConfig,
    _mark_job_failed,
    _mark_job_succeeded,
    create_job_queue,
    enqueue_job,
    list_job_tasks,
    update_job_task_progress,
)


def _queue(tmp_path):
    url = f"sqlite:///{(tmp_path / 'jobs.sqlite3').as_posix()}"
    return create_job_queue(JobQueueConfig(db_url=url))


def _payload():
    return {
        "tasks": [
            {"id": "prepare", "type": "controlnet.preprocess", "inputs": {"image": "@artifact:a"}},
            {"id": "render", "type": "sd15.text2img", "inputs": {"prompt": "test"}},
        ]
    }


def test_workflow_tasks_are_persisted_in_declared_order(tmp_path):
    engine, sessions, _worker = _queue(tmp_path)
    try:
        job, _ = enqueue_job(sessions, kind="workflow", payload=_payload())
        tasks = list_job_tasks(sessions, job.id)
        assert [(task.task_id, task.task_type, task.status) for task in tasks] == [
            ("prepare", "controlnet.preprocess", "queued"),
            ("render", "sd15.text2img", "queued"),
        ]
        assert [task.task_index for task in tasks] == [0, 1]
    finally:
        engine.dispose()


def test_task_progress_and_outputs_are_persisted(tmp_path):
    engine, sessions, _worker = _queue(tmp_path)
    try:
        job, _ = enqueue_job(sessions, kind="workflow", payload=_payload())
        update_job_task_progress(
            sessions, job.id, {"current_task": "prepare", "phase": "running"}
        )
        assert list_job_tasks(sessions, job.id)[0].status == "running"

        update_job_task_progress(
            sessions, job.id, {"current_task": "prepare", "phase": "completed_task"}
        )
        result = {
            "outputs": {"images": ["/outputs/result.png"]},
            "tasks": {
                "prepare": {"artifact": {"artifact_id": "p1"}},
                "render": {"images": ["/outputs/result.png"]},
            },
        }
        with sessions() as session:
            _mark_job_succeeded(session, job.id, result)

        tasks = list_job_tasks(sessions, job.id)
        assert [task.status for task in tasks] == ["succeeded", "succeeded"]
        assert tasks[1].output == {"images": ["/outputs/result.png"]}
        assert all(task.finished_at is not None for task in tasks)
    finally:
        engine.dispose()


def test_failure_marks_active_task_failed_and_remaining_tasks_skipped(tmp_path):
    engine, sessions, _worker = _queue(tmp_path)
    try:
        job, _ = enqueue_job(sessions, kind="workflow", payload=_payload())
        update_job_task_progress(
            sessions, job.id, {"current_task": "prepare", "phase": "running"}
        )
        with sessions() as session:
            _mark_job_failed(session, job.id, "preprocessor failed")
        tasks = list_job_tasks(sessions, job.id)
        assert [task.status for task in tasks] == ["failed", "skipped"]
        assert tasks[0].error == "preprocessor failed"
    finally:
        engine.dispose()
