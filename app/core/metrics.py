"""
Prometheus Metrics
==================
Central definition of all application-level metrics.
Import these counters/histograms/gauges wherever you need to record observations.
"""
from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "pitchmaster_requests_total",
    "Total HTTP requests by method, endpoint, and status code",
    ["method", "endpoint", "status"],
)

LLM_LATENCY = Histogram(
    "pitchmaster_llm_latency_seconds",
    "LLM chain invocation duration in seconds, labelled by agent type",
    ["agent"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

QUEUE_DEPTH = Gauge(
    "pitchmaster_queue_depth",
    "Current number of items waiting in the telemetry queue",
)
