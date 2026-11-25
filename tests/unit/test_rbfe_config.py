"""The configuration of a relative run (cufep.rbfe.config): its documented defaults and its refusals."""
from __future__ import annotations

import re

import pytest
import yaml

from cufep.rbfe.config import RbfeConfig


def _minimal(**more) -> dict:
    return {"system": {"protein": "p.pdb", "ligands": "l.sdf"}, **more}


def test_documented_defaults_are_the_defaults_of_the_code(repo):
    """docs/rbfe.md gives the defaults of the sections of a relative run; they are those of the code."""
    text = (repo / "docs" / "rbfe.md").read_text()
    block = yaml.safe_load(re.search(r"```yaml\n(.*?)```", text, re.S).group(1))
    defaults = RbfeConfig.model_validate(_minimal()).model_dump(mode="json")
    for section, values in block.items():
        for key, value in values.items():
            assert defaults[section][key] == value, f"{section}.{key}"
    for section in ("network", "mapping", "alchemy", "lambda_schedule"):
        assert block[section] == defaults[section], section


@pytest.mark.parametrize("more, message", [
    ({"sampling": {"gcmc": {"enabled": True}}}, "moves of the ABFE"),
    ({"membrane": {"enabled": True, "complex_leg": "water"}}, "complex_leg"),
    ({"network": {"edges": [["a", "a"]]}}, "to itself"),
    ({"network": {"edges": [["a", "b"], ["b", "a"]]}}, "twice"),
    ({"analysis": {"lrc": "reweight"}}, "lrc"),
])
def test_settings_a_relative_run_cannot_take_are_refused_when_read(more, message):
    with pytest.raises(ValueError, match=message):
        RbfeConfig.model_validate(_minimal(**more))
