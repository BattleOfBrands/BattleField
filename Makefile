############################## Makefile repo variables ##############################
VENV_NAME?=venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=${VENV_NAME}/bin/python

############################## Mandatory Targets for local development ##############################
.DEFAULT: help
help:
	@echo "make env"
	@echo "       Prepare development environment, use only once"
	@echo "make env_update"
	@echo "       If new dependencies are added, run this"
	@echo "make test_unit"
	@echo "       Run unit tests"
	@echo "make test_integration"
	@echo "       Run integration tests"
	@echo "make run"
	@echo "       Run project"
	@echo "make build_docker"
	@echo "       Create and run docker"

env:
	python3 -m venv $(VENV_NAME)
	$(VENV_ACTIVATE)
	${PYTHON} -m pip install -U pip
	${PYTHON} -m pip install -r requirements.txt
	touch $(VENV_NAME)/bin/activate

env_update:
	$(VENV_ACTIVATE)
	${PYTHON} -m pip install -r requirements.txt

test:
	$(VENV_ACTIVATE); $(PYTHON) -m pytest --log-cli-level=10 tests/ -v
	@echo "Unit tests done"

test_unit:
	$(VENV_ACTIVATE); $(PYTHON) -m pytest --log-cli-level=10 tests/unit/ -v
	@echo "Unit tests done"

test_integration:
	$(VENV_ACTIVATE); $(PYTHON) -m pytest --log-cli-level=10 tests/integration/ -v
	@echo "Integration tests done"

run:
	$(VENV_ACTIVATE); $(PYTHON) image_processor.py

build_docker:
	docker build -t logo_scout:latest .
	docker run -p 8000:8000 --rm logo_scout:latest