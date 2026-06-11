FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jdk-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Java 17 restricts reflective access that Spark + Arrow need (for
# mapInPandas, pandas_udf, etc). These --add-opens flags unblock them.
# JAVA_TOOL_OPTIONS is read by every JVM started in the container.
ENV JAVA_TOOL_OPTIONS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED"

WORKDIR /app

# Make /app importable so scripts under include/ can do `from utils...`
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the small English spaCy model as a direct wheel. More reproducible
# than `python -m spacy download` and works inside Docker build sandboxes
# that block the spaCy CLI's download flow.
# Swap URL to en_core_web_md or _trf later if accuracy on legal text is poor.
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Pre-download Legal-BERT into the image so containers start instantly. About
# 440 MB. Cached at /root/.cache/huggingface/hub/. Swap the model name if we
# upgrade to a different embedding model later.
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    name='nlpaueb/legal-bert-base-uncased'; \
    AutoTokenizer.from_pretrained(name); \
    AutoModel.from_pretrained(name)"

COPY . .

CMD ["/bin/bash"]
