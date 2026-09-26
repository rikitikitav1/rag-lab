# docling-serve with the models its image lacks, pinned by revision and checked by sha256
# v1.35.0, pinned by digest because a tag can be pushed again
FROM quay.io/docling-project/docling-serve-cu130@sha256:03b536b2d62811d41bebed3f91d654400fe834968e7c58649621117ba4705ee3

ARG MODELS=/opt/app-root/src/.cache/docling/models
ARG EASYOCR_URL=https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.1/cyrillic_g2.zip
ARG EASYOCR_SHA256=48d0f3b58f28aa64651ab1032cc2d498c4de25135829668e87c14e7a07529f29
ARG CODEFORMULA_REVISION=ecedbe111d15c2dc60bfd4a823cbe80127b58af4
ARG CODEFORMULA_SHA256=4b04e77af34c4e682a7ab1617628340d658f3c3dcd12456dd2a7fff805cf79d2

COPY docling/fetch_models.py /tmp/fetch_models.py
RUN python3 /tmp/fetch_models.py "$MODELS" "$EASYOCR_URL" "$EASYOCR_SHA256" \
        "$CODEFORMULA_REVISION" "$CODEFORMULA_SHA256"

# Tesseract ships English only; the Russian model is an OCR arm for the raster regime, tessdata tag 4.1.0
ARG TESS_RUS_URL=https://github.com/tesseract-ocr/tessdata/raw/4.1.0/rus.traineddata
ARG TESS_RUS_SHA256=681be2c2bead1bc7bd235df88c44e8e60ae73ae866840c0ad4e3b4c247bd37c2
USER 0
RUN curl -sfL -o /usr/share/tesseract/tessdata/rus.traineddata "$TESS_RUS_URL" && \
    echo "$TESS_RUS_SHA256  /usr/share/tesseract/tessdata/rus.traineddata" | sha256sum -c --quiet
USER 1001

COPY supervisor.py /opt/converter/supervisor.py
COPY docling/required_models.txt /opt/converter/required_models.txt
COPY docling/Dockerfile /opt/converter/Dockerfile
COPY stamp_build.py /tmp/stamp_build.py
# the copies above are root's, so the stamp is written as root and the image goes back to its own user
USER 0
RUN python3 /tmp/stamp_build.py /opt/converter/build.json docling \
        sha256:03b536b2d62811d41bebed3f91d654400fe834968e7c58649621117ba4705ee3
USER 1001
# one worker: two ran side by side and filled the 8 GB card on a whole book
ENV DOCLING_SERVE_ENG_LOC_NUM_WORKERS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CONVERTER_TOOL=docling \
    CONVERTER_CHILD="docling-serve run --host 127.0.0.1 --port 5002" \
    CONVERTER_CHILD_PORT=5002
CMD ["python3", "/opt/converter/supervisor.py"]
