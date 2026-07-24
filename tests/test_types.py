import dataclasses
import pytest

from src.core.types import GoalSpec, MainDimensions, ShipDesign


def test_goalspec_is_frozen():
    goal = GoalSpec(target_speed_ms=1.5, payload_kg=100.0, purpose="survey")
    with pytest.raises(dataclasses.FrozenInstanceError):
        goal.payload_kg = 200.0


def test_maindimensions_fields():
    dims = MainDimensions(loa=4.0, beam=1.3, depth=0.5, draft_design=0.3, cb=0.5)
    assert dims.loa == 4.0
    assert dims.cb == 0.5


def test_shipdesign_defaults():
    goal = GoalSpec(1.5, 100.0, "survey")
    dims = MainDimensions(4.0, 1.3, 0.5, 0.3, 0.5)
    design = ShipDesign(goal=goal, dims=dims)
    assert design.mesh_path is None
    assert design.extras == {}
