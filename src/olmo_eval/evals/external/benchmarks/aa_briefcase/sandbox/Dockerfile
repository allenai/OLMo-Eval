# Sandbox for AA-Briefcase-Lite: the document and scientific-computing stack the
# task prompt lists, an unprivileged `user` (UID 1000) with passwordless sudo, and
# the parser that prepares submissions for the judge.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca

ENV DEBIAN_FRONTEND=noninteractive
# Rootless container builds cannot drop privileges to apt's sandbox user.
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/99-disable-sandbox
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates chromium curl default-jre-headless ffmpeg fonts-dejavu \
        fonts-liberation ghostscript git imagemagick jq libpango-1.0-0 libpangoft2-1.0-0 \
        libreoffice-calc libreoffice-impress libreoffice-writer lmodern pandoc \
        poppler-utils sudo tesseract-ocr texlive-fonts-recommended texlive-latex-extra \
        texlive-latex-recommended texlive-pictures texlive-xetex \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/briefcase/requirements.txt
RUN uv pip install --system --no-cache -r /opt/briefcase/requirements.txt
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright
RUN python -m playwright install chromium && chmod -R a+rX /opt/playwright

COPY parse_submission.py /opt/briefcase/parse_submission.py

# Pythons uv downloads, including the one swe-rex's server runs on, go where `user`
# can execute them; uv's default under /root is closed to it.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN mkdir -p /opt/uv-python && chmod 755 /opt/uv-python

RUN useradd --uid 1000 --create-home --shell /bin/bash user \
    && echo 'user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/user
ENV MPLBACKEND=Agg
WORKDIR /home/user
USER user
