FROM python:3.12-slim

# This image is only for sandboxed execution of generated OpenAeroStruct code.
# It does not need Gemini or Ollama credentials because LLM calls stay in the
# host application process.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN useradd --create-home --shell /usr/sbin/nologin sandbox \
    && mkdir -p /workspace/src /workspace/openaerostruct_out \
    && chown -R sandbox:sandbox /workspace /home/sandbox

WORKDIR /workspace/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gfortran git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "numpy>=2.2.4" \
    scipy \
    "matplotlib>=3.10.1" \
    "pandas>=2.2.3" \
    "plotly>=6.0.1" \
    "openmdao>=3.38.0" \
    "python-dotenv>=1.1.1" \
    "openaerostruct @ git+https://github.com/ConanLee918/OpenAeroStruct.git@auto-plot-save"

USER sandbox
