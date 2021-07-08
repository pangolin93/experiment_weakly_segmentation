FROM continuumio/miniconda3

ARG PACKAGE="weakseg"

WORKDIR application

# Pre-install requirements, so that this sub-image is cached - no need to re-download it upon code-only change.
COPY environment.yml .
RUN conda env create -p conda_env -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-p", "./conda_env", "/bin/bash", "-c"]

# Prepare package installation copying sources
COPY setup.py .
COPY weakseg weakseg/

# Install the package
RUN pip install -e .

# Run it
ENTRYPOINT ["conda", "run", "--no-capture-output", "-p", "./conda_env", "python", "weakseg/main.py"]
