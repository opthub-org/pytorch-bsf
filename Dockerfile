FROM condaforge/miniforge3:26.7.2-0

# Install pytorch-bsf with PyTorch from the conda-forge channel only
# (Miniforge does not configure the Anaconda `defaults` channel).
# On x86_64, avoid mkl 2026.2.0, which lacks libmkl_gf_lp64.so
# (https://github.com/conda-forge/intel_repack-feedstock/issues/145).
WORKDIR /opt/pytorch-bsf
COPY environment.yml pyproject.toml ./
COPY torch_bsf ./torch_bsf
RUN grep -v '^\s*- -e \.' environment.yml > /tmp/env.yml && \
    if [ "$(uname -m)" = x86_64 ]; then echo '- "mkl<2026.2"' >> /tmp/env.yml; fi && \
    conda env update -n base -f /tmp/env.yml && \
    pip install --no-cache-dir --no-deps . && \
    conda clean -afy

WORKDIR /workspace
