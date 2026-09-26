# MinerU 4 on the vLLM image its own Dockerfile names, with the standard tier's models baked in
# v0.21.0, pinned by digest because a tag can be pushed again
FROM vllm/vllm-openai:v0.21.0@sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9

RUN apt-get update && apt-get install -y libgl1 && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install 'mineru[torch]==4.0.7' --break-system-packages && python3 -m pip cache purge

ENV MINERU_MODEL_SMALL_BACKEND=torch \
    MINERU_MODEL_VLM_ENGINE=vllm
RUN mineru-kit models download --tier standard --small-backend torch --vlm-engine vllm -s huggingface
# the vendor publishes no hashes: these are the files we baked, hashed on the first download of 24.09.2026,
# MinerU-4_models_torch at 2b3afb86f4d23fa623f6b2f4b2279ed4ef93a89d, MinerU2.5-Pro-2605-1.2B at bff20d4ae2bf202df9f45284b4d43681555a97ed
COPY mineru/models.sha256 /opt/converter/models.sha256
RUN if [ -s /opt/converter/models.sha256 ]; then sha256sum -c --quiet /opt/converter/models.sha256; fi

COPY supervisor.py /opt/converter/supervisor.py
COPY mineru/required_models.txt /opt/converter/required_models.txt
COPY mineru/Dockerfile /opt/converter/Dockerfile
COPY stamp_build.py /tmp/stamp_build.py
RUN python3 /tmp/stamp_build.py /opt/converter/build.json mineru \
        sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9
# offline at run time: a missing model fails loud instead of downloading behind the stamp's back
ENV MINERU_MODEL_SOURCE=local \
    CONVERTER_TOOL=mineru \
    CONVERTER_CHILD="mineru-kit api-server --host 127.0.0.1 --port 5002 --tier standard" \
    CONVERTER_CHILD_PORT=5002 \
    CONVERTER_CHILD_HEALTH=/v1/health
ENTRYPOINT []
CMD ["python3", "/opt/converter/supervisor.py"]
