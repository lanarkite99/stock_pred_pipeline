# Commands

## Health

Health check:

```bash
curl http://localhost:8000/health
```

## Training

Train Parent:

```bash
curl -X POST http://localhost:8000/train-parent
```

Train Child (AAPL):

```bash
curl -X POST http://localhost:8000/train-child -H "Content-Type: application/json" -d '{"ticker":"AAPL"}'
```

Train Child (INFY.NS):

```bash
curl -X POST http://localhost:8000/train-child -H "Content-Type: application/json" -d '{"ticker":"INFY.NS"}'
```

## Status

Task Status (parent):

```bash
curl http://localhost:8000/status/parent_training
```

Task Status (child):

```bash
curl http://localhost:8000/status/infy.ns
```

## Prediction

Predict Parent:

```bash
curl -X POST http://localhost:8000/predict-parent
```

Predict Child (AAPL):

```bash
curl -X POST http://localhost:8000/predict-child -H "Content-Type: application/json" -d '{"ticker":"AAPL"}'
```

Predict Child (HDFCBANK.NS):

```bash
curl -X POST http://localhost:8000/predict-child -H "Content-Type: application/json" -d '{"ticker":"HDFCBANK.NS"}'
```

## Analysis

Analyze (AAPL):

```bash
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"ticker":"AAPL"}'
```

Analyze (INFY.NS):

```bash
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"ticker":"INFY.NS"}'
```

Analyze again to hit semantic cache:

```bash
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"ticker":"INFY.NS"}'
```

## Monitoring

Monitor (AAPL):

```bash
curl -X POST http://localhost:8000/monitor/AAPL
```

Monitor (INFY.NS):

```bash
curl -X POST http://localhost:8000/monitor/INFY.NS
```

## Metrics

Prometheus metrics endpoint:

```bash
curl http://localhost:8000/metrics
```

## URLs

- FastAPI: `http://localhost:8000`
- FastAPI Docs: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Streamlit App: `http://localhost:8502`
- Redis Stack UI: `http://localhost:8001`
- DagsHub Repo: `https://dagshub.com/meetapple191/stock-pred-end2end-agentic`
