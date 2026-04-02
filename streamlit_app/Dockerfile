FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY streamlit_app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app/app.py ./app.py

EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
