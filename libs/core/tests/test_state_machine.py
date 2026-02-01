from libs.core import models, state_machine


def test_valid_task_transition():
    assert state_machine.validate_task_transition(
        models.TaskStatus.pending, models.TaskStatus.ready
    )


def test_invalid_task_transition():
    assert not state_machine.validate_task_transition(
        models.TaskStatus.pending, models.TaskStatus.completed
    )
