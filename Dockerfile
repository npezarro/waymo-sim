FROM python:3.11-slim

WORKDIR /app

# System deps for pygame (headless)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 \
        libsdl2-ttf-2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Pin NumPy + BLAS for deterministic floating-point
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "gymnasium>=0.29.0" \
    "pyyaml>=6.0" \
    "pygame>=2.5.0" \
    "pytest>=7.0" \
    "pytest-cov>=4.0"

COPY . .

# Verify tests pass inside the container
RUN python -m pytest tests/ -v

ENV SDL_VIDEODRIVER=dummy
ENV PYTHONPATH=/app

CMD ["python", "-m", "pytest", "tests/", "-v"]
