FROM condaforge/miniforge3:26.9.4-0

# Install pytorch-bsf with PyTorch from the conda-forge channel only
# (Miniforge does not configure the Anaconda `defaults` channel).
WORKDIR /opt/pytorch-bsf
# README.md and LICENSE are part of the package metadata (readme and license-files in pyproject.toml).
COPY environment.yml pyproject.toml README.md LICENSE ./
COPY torch_bsf ./torch_bsf
RUN grep -v '^\s*- -e \.' environment.yml > /tmp/env.yml && \
    conda env update -n base -f /tmp/env.yml && \
    pip install --no-cache-dir --no-deps . && \
    conda clean -afy

WORKDIR /workspace
