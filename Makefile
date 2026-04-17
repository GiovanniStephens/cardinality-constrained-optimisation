.PHONY: install test build-cpp clean

install:
	pip install -e ".[dev]"

test:
	python -m unittest discover tests

build-cpp:
	@echo "Building C++ optimiser..."
	cd cpp && mkdir -p build && cd build && cmake .. && make -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu)
	cp cpp/build/optimisation cpp/optimisation
	@echo "Binary ready at cpp/optimisation"

clean:
	rm -rf cpp/build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info build dist
