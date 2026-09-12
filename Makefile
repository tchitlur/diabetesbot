PYTHON ?= python
NPM ?= npm

.PHONY: setup fixtures cohort fit validate test api web demo

setup:
	$(PYTHON) -m pip install -r requirements.txt
	-@if [ -f web/package.json ]; then cd web && $(NPM) install; fi

fixtures:
	$(PYTHON) scripts/make_fixtures.py

cohort:
	-$(PYTHON) scripts/build_cohort.py

fit:
	-$(PYTHON) scripts/fit_demo.py

validate:
	-$(PYTHON) tests/validation/run_all.py

test:
	$(PYTHON) -m pytest -q

api:
	-$(PYTHON) -m uvicorn api.main:app --reload --port 8000

web:
	-cd web && $(NPM) run dev

demo: fixtures
	-@if [ ! -d data/synthetic/patient_02 ]; then $(MAKE) cohort; fi
	-@$(MAKE) fit
	-@$(MAKE) -j2 api web
