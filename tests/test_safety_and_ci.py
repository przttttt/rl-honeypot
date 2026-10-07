import numpy as np
import pytest

from rlhoneypot.metrics.ci import (auroc_ap, brier_score, bootstrap_ci, precision_at_k, wilson_interval)
from rlhoneypot.safety.shim import (INSULT_MESSAGES, SafeAgentWrapper, SandboxViolation, insult_for, require_sandbox)

class _FixedAgent:
    def __init__(self, action):
        self.action = action

    def act(self, state, context=None):
        return self.action

def test_allow_filtered_in_compromise_states():
    w = SafeAgentWrapper(_FixedAgent(0))  
    assert w.act(4) == 0                  
    assert w.act(5) == 1                  
    assert w.act(6) == 1
    assert w.act(7) == 1
    assert w.violation_counts["allow_filtered"] == 3

def test_insult_cap_enforced_per_session():
    w = SafeAgentWrapper(_FixedAgent(3), insult_cap=2)
    assert w.act(0) == 3
    assert w.act(0) == 3
    assert w.act(0) == 2                  # third insult → delay
    assert w.violation_counts["insult_filtered"] == 1
    w.reset_session()
    assert w.act(0) == 3

def test_disabled_wrapper_passes_through():
    w = SafeAgentWrapper(_FixedAgent(0), enabled=False)
    assert w.act(5) == 0

def test_sandbox_guard():
    import os
    old = os.environ.get("rlhoneypot_SANDBOX")
    try:
        os.environ["rlhoneypot_SANDBOX"] = "0"
        with pytest.raises(SandboxViolation):
            require_sandbox()
        os.environ["rlhoneypot_SANDBOX"] = "1"
        require_sandbox()
    finally:
        if old is None:
            os.environ.pop("rlhoneypot_SANDBOX", None)
        else:
            os.environ["rlhoneypot_SANDBOX"] = old

def test_insult_messages_match_paper_and_rendering():
    assert "won't change your chances" in INSULT_MESSAGES["chmod"]
    assert insult_for("chmod 777 /tmp") == INSULT_MESSAGES["chmod"]
    assert insult_for("sudo rm -rf /") == INSULT_MESSAGES["deletion"]
    assert insult_for("top") == INSULT_MESSAGES["top"]
    assert insult_for("free -m") == INSULT_MESSAGES["resource_query"]

def test_wilson_known_values():
    lo, hi = wilson_interval(5, 50)
    assert lo < 0.1 and 0.2 < hi < 0.35   
    assert lo < 0.1 < hi
    lo, hi = wilson_interval(0, 100)
    assert lo <= 1e-12 and hi < 0.04       
    lo0, hi0 = wilson_interval(0, 0)
    import math
    assert math.isnan(lo0) and math.isnan(hi0)

def test_bootstrap_ci_deterministic_and_sane():
    vals = np.arange(100.0)
    lo1, hi1 = bootstrap_ci(vals, seed=7)
    lo2, hi2 = bootstrap_ci(vals, seed=7)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 < 49.5 < hi1

def test_auroc_perfect_and_reversed():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc_ap(y, s)["auroc"] == 1.0
    assert auroc_ap(y, -s)["auroc"] == 0.0

def test_auroc_ties_average_ranks():
    y = np.array([0, 1])
    s = np.array([0.5, 0.5])
    assert auroc_ap(y, s)["auroc"] == 0.5

def test_precision_at_k_and_brier():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert precision_at_k(y, s, 2) == 0.5
    assert brier_score(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
 