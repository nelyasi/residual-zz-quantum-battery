PYTHON ?= python

.PHONY: install verify results audit figures test all clean checksums

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

verify:
	$(PYTHON) scripts/verify_archive.py
	shasum -a 256 -c SHA256SUMS --quiet

results:
	$(PYTHON) scripts/reproduce_results.py --check

audit:
	$(PYTHON) scripts/audit_measurements.py

figures:
	$(PYTHON) scripts/make_figures.py build/figures

test:
	$(PYTHON) -m pytest

all: verify results audit figures test

checksums:
	find data results figures -type f ! -name '.*' | LC_ALL=C sort | xargs shasum -a 256 > SHA256SUMS

clean:
	rm -rf build .pytest_cache
