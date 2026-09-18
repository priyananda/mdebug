# The mdebug inference server on Cloud Run.
#
# Single stage on purpose: every dependency ships a manylinux wheel for CPython
# 3.12, so there is no compiler to leave behind and a builder stage would save
# only pip/setuptools against a ~1.3 GB image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Matches run.py's os.chdir(HERE), so `app.main` imports the same way it does
# locally and there is one import contract rather than two.
WORKDIR /srv/server

# torch first and alone: biggest, slowest, least likely to change.
# --index-url, NOT --extra-index-url -- there must be no resolution path by
# which the ~900 MB CUDA wheel and its nvidia-* dependencies can be chosen.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.14.0+cpu

COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Weights get their own layer, ABOVE the code. Layers invalidate forward, and
# code changes every deploy while weights change almost never -- this way a
# normal deploy re-pushes a megabyte instead of 342 of them.
#
# The test fails the BUILD if .gcloudignore filtered the file out of the
# upload, minutes before a deploy would have served random weights instead.
COPY checkpoints/ckpt_5000.weights.pt ./checkpoints/ckpt_5000.weights.pt
RUN test -s checkpoints/ckpt_5000.weights.pt

COPY data/tokenizer/ ./data/tokenizer/

# Code last: the only layer a normal deploy actually changes.
COPY run.py ./
COPY src/ ./src/
COPY app/ ./app/

# MDEBUG_CHECKPOINT is not optional here: runtime.py defaults to ckpt_5000.pt
# but the image carries the stripped ckpt_5000.weights.pt. Both paths are read
# at import time, which ENV satisfies.
#
# OMP/MKL_NUM_THREADS and MDEBUG_TORCH_THREADS are both needed: the OpenMP
# runtime reads its variables at library init, before Python can call
# torch.set_num_threads. Keep all three equal to the deployed --cpu.
ENV MDEBUG_HOST=0.0.0.0 \
    PORT=8080 \
    MDEBUG_CHECKPOINT=/srv/server/checkpoints/ckpt_5000.weights.pt \
    MDEBUG_TOKENIZER_DIR=/srv/server/data/tokenizer \
    MDEBUG_REQUIRE_CHECKPOINT=1 \
    MDEBUG_TORCH_THREADS=2 \
    MDEBUG_EXECUTOR_WORKERS=2 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN useradd --create-home --uid 1001 app && chown -R app:app /srv/server
USER app

EXPOSE 8080
CMD ["python", "run.py"]
