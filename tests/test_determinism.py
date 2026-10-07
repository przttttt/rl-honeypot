import json
from pathlib import Path

import numpy as np
import pytest

from rlhoneypot.agents.dqn import DQNAgent
from rlhoneypot.agents.tabular_q import TabularQAgent
from rlhoneypot.corpus.generator import generate_corpus
from rlhoneypot.envs import ResponseEnv
from rlhoneypot.pipeline.label import label_events

@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    d1 = tmp_path_factory.mktemp("c1")
    d2 = tmp_path_factory.mktemp("c2")
    return d1, d2

def test_corpus_byte_identical(tiny_corpus):
    d1, d2 = tiny_corpus
    p1 = generate_corpus(d1, n_phase1=200, n_phase2=150, seed=99)
    p2 = generate_corpus(d2, n_phase1=200, n_phase2=150, seed=99)
    assert p1.phase1.read_bytes() == p2.phase1.read_bytes()
    assert p1.ground_truth.read_bytes() == p2.ground_truth.read_bytes()

def test_labels_deterministic(tiny_corpus):
    d1, _ = tiny_corpus
    p1 = generate_corpus(d1, n_phase1=200, n_phase2=150, seed=99)
    from rlhoneypot.cowrie.parser import iter_cowrie_events
    labels = label_events(list(iter_cowrie_events(p1.phase1)))
    sig = [(l.session_id, len(l.states_actions), l.n_unmapped) for l in labels]
    labels2 = label_events(list(iter_cowrie_events(p1.phase1)))
    sig2 = [(l.session_id, len(l.states_actions), l.n_unmapped) for l in labels2]
    assert sig == sig2

def test_tabular_curve_identical_across_runs(tiny_corpus):
    d1, _ = tiny_corpus
    p1 = generate_corpus(d1, n_phase1=200, n_phase2=150, seed=99)
    from rlhoneypot.cowrie.parser import iter_cowrie_events
    from rlhoneypot.transitions.estimate import estimate
    est = estimate([ls.transitions for ls in label_events(list(iter_cowrie_events(p1.phase1)))])
    curves = []
    for _ in range(2):
        env = ResponseEnv(est, max_steps=20, seed=5)
        agent = TabularQAgent(seed=5, alpha=0.2, gamma=0.9, epsilon=0.2)
        rng = np.random.default_rng(5)
        curve = []
        for ep in range(40):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done = trunc = False
            s_prev = int(obs)
            tot = 0.0
            while not (done or trunc):
                a = agent.act(s_prev)
                obs2, r, done, trunc, info = env.step(a)
                agent.update(s_prev, a, r, int(obs2), done or trunc)
                tot += r
                s_prev = int(obs2)
            curve.append(tot)
        curves.append(curve)
    assert curves[0] == curves[1]

def test_dqn_curve_identical_on_cpu(tiny_corpus):
    d1, _ = tiny_corpus
    p1 = generate_corpus(d1, n_phase1=200, n_phase2=150, seed=99)
    from rlhoneypot.cowrie.parser import iter_cowrie_events
    from rlhoneypot.transitions.estimate import estimate
    est = estimate([ls.transitions for ls in label_events(list(iter_cowrie_events(p1.phase1)))])
    from rlhoneypot.envs import ContextualResponseEnv

    curves = []
    for run_i in range(2):
        agent = DQNAgent(obs_dim=11, n_actions=5,
                         cfg={"hidden": (16, 16), "lr": 1e-3, "batch_size": 16,
                              "replay_capacity": 500, "warmup_steps": 20,
                              "target_sync": 10, "gamma": 0.9, "warm_start_bc_steps": 5},
                         seed=17 + run_i * 0)  # same seed both runs
        env = ContextualResponseEnv(est, max_steps=8, seed=17)
        rng = np.random.default_rng(17)
        curve = []
        for ep in range(6):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done = trunc = False
            ep_ret = 0.0
            while not (done or trunc):
                a = agent.act(context=np.asarray(obs))
                obs2, r, done, trunc, info = env.step(a)
                agent.observe(obs, a, r, obs2, done or trunc)
                obs = obs2
                ep_ret += r
            curve.append(ep_ret)
        curves.append(curve)
    assert curves[0] == curves[1]

def test_json_canonical_writing(tmp_path):
    from rlhoneypot.logging_util import write_json
    obj = {"b": 1, "a": [3.0, {"z": None, "y": np.float64(2.5)}]}
    f1 = tmp_path / "x.json"
    f2 = tmp_path / "y.json"
    write_json(f1, obj)
    write_json(f2, obj)
    assert f1.read_bytes() == f2.read_bytes()
    assert f1.read_text().endswith("\n")
