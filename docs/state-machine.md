# State Machine

Job states: queued, planning, running, succeeded, failed, canceled.
Task states: pending, ready, running, blocked, completed, accepted, rework_requested, failed, canceled.
Transitions are enforced in libs/core/state_machine.py.
