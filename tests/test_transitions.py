import numpy as np

from rlhoneypot.transitions.estimate import estimate, heldout_logloss, label_session
from rlhoneypot.pipeline.label import LabelledSession

def test_label_session_collapses_consecutive_states():
    sa = [(0, 0), (0, 0), (1, 1), (1, 1), (3, 2)]
    tr = label_session(sa)
    assert tr == [(0, 0, 1), (1, 1, 3)]

def test_label_session_drops_self_loops_only():
    assert label_session([(5, 5), (5, 5)]) == []
    assert label_session([]) == []

def test_laplace_smoothing_gives_nonzero_on_supported_rows():
    sessions = [[(0, 0, 1), (1, 1, 7)]]
    est = estimate(sessions, laplace_alpha=1.0)
    for s in range(8):
        for a in range(7):
            if est.support_mask[s, a]:
                assert (est.probs[s, a] > 0).all()
                assert abs(est.probs[s, a].sum() - 1.0) < 1e-9
    for s in range(8):
        for a in range(7):
            if not est.support_mask[s, a]:
                assert (est.probs[s, a] == 0).all()

def test_unvisited_row_uniform():
    est = estimate([[(0, 0, 1)]], laplace_alpha=1.0)
    assert np.allclose(est.probs[3, 2], 1 / 8, atol=1e-9)
    assert (3, 2, 0) in [tuple(x) for x in est.low_confidence_rows]

def test_counts_match_transitions_and_out_of_support():
    est = estimate([[(0, 0, 1), (1, 1, 7), (7, 6, 7)]])
    assert est.out_of_support_counts == 0
    assert est.counts[7, 6, 7] == 1
    assert est.counts[0, 0, 1] == 1
    assert est.counts[1, 1, 7] == 1
    assert est.counts.sum() == 3
    est2 = estimate([[(2, 3, 4)]])
    assert est2.out_of_support_counts == 1
    assert est2.counts.sum() == 0
    from rlhoneypot.transitions.estimate import label_session
    tr = label_session([(0, 0), (1, 1), (7, 6), (6, 3)])
    assert tr == [(0, 0, 1), (1, 1, 7), (7, 6, 6)]
    est3 = estimate([tr])
    assert est3.counts.sum() == 3
    assert est3.out_of_support_counts == 0

def test_heldout_logloss_beats_uniform_on_informative_data():
    train = estimate([[(0, 0, 1)] * 40 + [(0, 0, 3)] * 10])
    eval_sess = [[(0, 0, 1)] * 5]
    out = heldout_logloss(train, eval_sess)
    assert out["improvement_nats"] > 0
