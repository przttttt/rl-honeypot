.PHONY: setup test reproduce smoke clean report figures

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -q -r requirements.txt
	.venv/bin/pip install -q .
	@echo "Setup complete. Next: make test"

test:
	rlhoneypot_SANDBOX=1 .venv/bin/pytest -q tests/

smoke:
	rlhoneypot_SANDBOX=1 .venv/bin/python -m rlhoneypot.repro.run_all --out results --smoke

reproduce:
	rlhoneypot_SANDBOX=1 .venv/bin/python -m rlhoneypot.repro.run_all --out results

report:
	rlhoneypot_SANDBOX=1 .venv/bin/python -c "from rlhoneypot.repro.report import generate_report; generate_report('results')"

figures:
	.venv/bin/python -m rlhoneypot.repro.figures

clean:
	rm -rf results data/raw data/siem_bulk_*.ndjson
