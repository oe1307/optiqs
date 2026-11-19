ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim AS python-builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml ./
COPY optiqs/ ./optiqs/

RUN python -m pip install --only-binary=:all: --no-compile .

CMD ["python", "-c", "import sys; import numpy, scipy, optiqs as opt; p = opt.pnrd(opt.coherent(20, alpha=1.0)); assert abs(float(p.sum()) - 1.0) < 1e-10; print(f'PASS: Python {sys.version.split()[0]}, NumPy {numpy.__version__}, SciPy {scipy.__version__}')"]
