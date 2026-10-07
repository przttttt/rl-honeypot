import numpy as np
import pytest

from rlhoneypot.agents.tabular_q import TabularQAgent
from rlhoneypot.cowrie.parser import iter_cowrie_events
from rlhoneypot.corpus.generator import generate_corpus
from rlhoneypot.envs import ContextualResponseEnv, ResponseEnv
from rlhoneypot.opelogs.logger import OPELog, log_randomised_policy, support_check
from rlhoneypot.ope.estimators import doubly_robust, ips, snips
from rlhoneypot.pipeline.label import label_events
from rlhoneypot.siem.index import (OfflineIndex, WazuhRules, correlation_lift,siem_alert_stream)
from rlhoneypot.transitions.estimate import estimate, heldout_logloss

@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("int")
    return generate_corpus(d, n_phase1=400, n_phase2=300, seed=123)

@pytest.fixture()
def est():
    sessions = []
    for _ in range(50):
        sessions.append([(0, 0, 1)] * 3 + [(1, 1, 7)])
    for _ in range(20):
        sessions.append([(0, 0, 2)] + [(2, 2, 4)])
    return estimate(sessions)

def test_full_pipeline(corpus):
    events = list(iter_cowrie_events(corpus.phase1))
    assert len(events) > 400
    labels = label_events(events)
    assert len(labels) == 400
    tr = [l.transitions for l in labels]
    assert tr
    est = estimate(tr)
    assert est.counts.sum() > 100
    assert est.out_of_support_counts == 0
    env = ResponseEnv(est, seed=1)
    obs, _ = env.reset(seed=1)
    done = trunc = False
    while not (done or trunc):
        obs, r, done, trunc, info = env.step(1)  
    assert isinstance(r, float)
    env = ResponseEnv(est, seed=1)
    obs, _ = env.reset(seed=1)
    done = trunc = False
    while not (done or trunc):
        obs, r, done, trunc, info = env.step(1) 
    assert isinstance(r, float)

def test_ope_estimators_sane(corpus):
    events = list(iter_cowrie_events(corpus.phase2))
    labels = [l for l in label_events(events) if l.transitions][:60]
    est = estimate([l.transitions for l in labels])
    agent = TabularQAgent(seed=0)
    env = ResponseEnv(est, seed=0)
    rng = np.random.default_rng(0)
    for ep in range(80):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = trunc = False
        s_prev = int(obs)
        while not (done or trunc):
            a = agent.act(s_prev)
            obs2, r, done, trunc, info = env.step(a)
            agent.update(s_prev, a, r, int(obs2), done or trunc)
            s_prev = int(obs2)
    log = log_randomised_policy([l.states_actions for l in labels], est, agent.q,
                                epsilon=0.3, rng=np.random.default_rng(1))
    assert support_check(log, 5, 5)["support_ok"]
    pi = np.zeros((8, 5))
    for s in range(8):
        pi[s, agent.policy()[s]] = 1.0
    log_dict = {"states": log.states, "actions": log.actions,
                "propensities": log.propensities, "rewards": log.rewards,
                "next_states": log.next_states, "dones": log.dones,
                "sessions": log.sessions, "q_values": log.q_values,
                "v_values": log.state_values}
    i = ips(log_dict, pi)
    s = snips(log_dict, pi)
    d = doubly_robust(log_dict, pi)
    for est_out in (i, s, d):
        assert np.isfinite(est_out["estimate"])
        assert est_out["ci"][0] <= est_out["estimate"] <= est_out["ci"][1]

def test_siem_correlation_pipeline(corpus):
    events = list(iter_cowrie_events(corpus.phase2))
    labels = label_events(events)
    index = OfflineIndex()
    for ls in labels:
        for s, a in ls.states_actions:
            index.add({"@timestamp": "", "session": {"id": ls.session_id},
                       "rlhoneypot": {"mdp_state": s, "mdp_action": a}})
    cfg_rules = {"wazuh": {"rule_id_base": 100100, "rules": [
        {"id_offset": s, "name": f"r{s}", "mapped_state": s, "level": 3 + s} for s in range(8)]}}
    rules = WazuhRules.from_config(cfg_rules)
    rng = np.random.default_rng(0)
    alerts = siem_alert_stream(index, rules, [0.05] * 8, noise_rate=0.1, rng=rng)
    assert alerts
    counts = [0] * 8
    for a in alerts:
        counts[a["rlhoneypot"]["mdp_state"]] += 1
    lift = correlation_lift(index.count_by_state(), counts, 0.1)
    assert all(s["alert_rate"] is not None for s in lift["states"])
    assert all(0 <= s["alert_rate_ci95"][0] <= s["alert_rate_ci95"][1] <= 1
               for s in lift["states"])

def test_contextual_env_shapes(est):
    env = ContextualResponseEnv(est, max_steps=6, seed=2)
    obs, _ = env.reset(seed=2)
    assert obs.shape == (11,)
    obs2, r, done, trunc, info = env.step(1)
    assert obs2.shape == (11,)
