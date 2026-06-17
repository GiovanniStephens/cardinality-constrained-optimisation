.PHONY: install test build-cpp clean refresh refresh-incremental retry-core health-check

# Concurrency for downloads (capped to PIPELINE_MAX_WORKERS in config).
WORKERS ?= 24

install:
	pip install -e ".[dev]"

test:
	python -m unittest discover tests

# ── Data refresh (see docs/DATA_REFRESH.md) ─────────────────────────────────
# Full-universe refresh: all equities + ETFs, full history. Auto-runs
# post-promotion validation on production and a final health check that
# verifies the liquid core actually landed. Hours (Yahoo throttles the proxy).
refresh:
	python -m src.download --asset-types equities etfs --workers $(WORKERS)

# Faster catch-up: only dates newer than the latest already in the DB.
# (Now promotes correctly — incremental staging skips the min_history gate.)
refresh-incremental:
	python -m src.download --asset-types equities etfs --incremental --workers $(WORKERS)

# Recovery: force-refresh the protected core watchlist (data/core_etfs.csv),
# bypassing the bad-ticker cache. Run this if `health-check` flags stale core.
retry-core:
	python -m src.download --from-csv data/core_etfs.csv --asset-type etf \
		--ignore-bad-cache --workers $(WORKERS)

# Verify the liquid core is current and active, without downloading.
health-check:
	python -m src.download --health-check

build-cpp:
	@echo "Building C++ optimiser..."
	cd cpp && mkdir -p build && cd build && cmake .. && make -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu)
	cp cpp/build/optimisation cpp/optimisation
	@echo "Binary ready at cpp/optimisation"

clean:
	rm -rf cpp/build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info build dist
