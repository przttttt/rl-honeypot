import numpy as np

from rlhoneypot.mdp.rewards import (COMPROMISE_STATES, HARM_BY_NEXT_STATE, reward, reward_table)

def test_harm_matches_paper_table3_tiers():
    assert HARM_BY_NEXT_STATE[0] == 1.0
    assert HARM_BY_NEXT_STATE[1] == 2.0
    for s in (2, 3, 4, 5, 6, 7):
        assert HARM_BY_NEXT_STATE[s] == 5.0

def test_allow_is_bad_in_compromise_states_good_in_benign():
    r_allow_S5 = reward(4, 0, 5)   
    r_block_S5 = reward(4, 1, 5)
    assert r_allow_S5 < 0 < r_block_S5
    r_allow_S0 = reward(0, 0, 1)   
    assert r_allow_S0 > 0

def test_terminate_ends_with_cost_and_positive_on_compromise():
    assert reward(4, 4, 6) > 0
    assert reward(0, 4, 1) < 0  

def test_reward_table_shape_and_determinism():
    t1, t2 = reward_table(), reward_table()
    assert t1.shape == (8, 5, 8)
    assert np.array_equal(t1, t2)

def test_no_degenerate_allowing_policy():
    t = reward_table()
    always_allow_value = 0.0
    best_response_value = 0.0
    for s in range(8):
        for s2 in range(8):
            always_allow_value += t[s, 0, s2]
            best_response_value += t[s, :, s2].max()
    assert best_response_value > always_allow_value
