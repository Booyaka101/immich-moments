# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

# ffmpeg does every frame grab and audio extraction; libgl/libglib are OpenCV's runtime,
# which PySceneDetect loads to decode.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# "cuda" pulls the cuBLAS and cuDNN wheels faster-whisper needs on a GPU box. The default
# build is CPU only, because those wheels add about 900 MB to the image.
ARG EXTRAS=""

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}" && rm -rf /src

ENV DATA_DIR=/data \
    HF_HOME=/data/models \
    IMMICH_ML_URL=http://immich-machine-learning:3003 \
    IMMICH_MOMENTS_HOST=0.0.0.0 \
    IMMICH_MOMENTS_PORT=8099

RUN useradd --create-home --uid 1000 moments && mkdir -p /data && chown moments:moments /data
USER moments
VOLUME ["/data"]
EXPOSE 8099

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/api/stats' % os.environ['IMMICH_MOMENTS_PORT'], timeout=4).status == 200 else 1)"

ENTRYPOINT ["immich-moments"]
CMD ["serve"]
