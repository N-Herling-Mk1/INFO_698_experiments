# FORGE — Docker convenience targets.
# EXP selects the experiment (default genre); PORT overrides the host port.
EXP  ?= genre
PORT ?= 5000

.PHONY: build up lab eda down logs native help

help:
	@echo "FORGE / docker targets"
	@echo "  make build              build the image (CPU)"
	@echo "  make up   [EXP=genre]   dashboard -> http://localhost:$(PORT)"
	@echo "  make lab                interactive shell (EDA / training)"
	@echo "  make eda  [EXP=genre]   run that experiment's EDA inside the container"
	@echo "  make down               stop everything"
	@echo "  make native             run the dashboard WITHOUT docker (needs flask)"

build:
	@echo "[forge] building image (CPU)…"
	docker compose build

up:
	@echo "[forge] dashboard for '$(EXP)' -> http://localhost:$(PORT)  (Ctrl-C to stop)"
	EXP=$(EXP) PORT=$(PORT) docker compose up dashboard

lab:
	@echo "[forge] interactive lab shell — repo mounted at /workspace, env at /opt/forge-venv"
	docker compose run --rm lab

eda:
	@echo "[forge] running EDA for '$(EXP)' (override DATA + PHASE as needed)"
	docker compose run --rm lab \
	  python projects/$(EXP)/eda/run_eda.py --phase $(PHASE) --data-root $(DATA)

down:
	@echo "[forge] stopping…"
	docker compose down

logs:
	docker compose logs -f dashboard

native:
	@echo "[forge] native run (no docker) -> http://127.0.0.1:5000"
	python projects/$(EXP)/app/server.py
