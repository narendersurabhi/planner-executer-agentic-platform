from libs.core import models, state_machine


def test_valid_task_transition():
    assert state_machine.validate_task_transition(
        models.TaskStatus.pending, models.TaskStatus.ready
    )


def test_invalid_task_transition():
    assert not state_machine.validate_task_transition(
        models.TaskStatus.pending, models.TaskStatus.completed
    )


def test_valid_job_transition_from_queued_to_failed():
    assert state_machine.validate_job_transition(
        models.JobStatus.queued, models.JobStatus.failed
    )


def test_invalid_job_transition_from_succeeded_to_failed():
    assert not state_machine.validate_job_transition(
        models.JobStatus.succeeded, models.JobStatus.failed
    )
