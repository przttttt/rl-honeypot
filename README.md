# Problem Statement

Adaptive honeypots built on Q-learning like Q-Cowrie choose responses to
attacker commands from hand discretized state spaces and are evaluated mainly on
internal metrics. Their decisions have not been checked against SIEM correlation
analytics the tools security teams use to determine attacker behavior so the
reported gains might not show real world deception value.

Deep reinforcement learning could offer detailed session context but it has not
been compared with tabular methods in this area. Since live attackers cannot be
tested freely any comparison must rely on policy evaluation, which depends on
how well the logged data covers the actions a new policy would take. It is still
unclear whether a DQN agent that includes SIEM-derived features makes responses
than Q-Cowrie.
