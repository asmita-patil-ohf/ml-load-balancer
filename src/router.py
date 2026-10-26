
from dataclasses import dataclass
from typing import Optional


@dataclass
class BackendServer:
    server_id: str
    url: str
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    active_requests: int = 0
    queue_length: int = 0
    recent_avg_latency_ms: float = 50.0
    is_healthy: bool = True

    @property
    def available(self) -> bool:
        return self.is_healthy


class RoundRobinRouter:
    def __init__(self):
        self.next_index = 0

    def choose(self, servers):
        healthy = [s for s in servers if s.available]

        if not healthy:
            raise RuntimeError("No healthy backend servers")

        # Preserve the index across requests.
        selected = healthy[self.next_index % len(healthy)]
        self.next_index = (self.next_index + 1) % len(healthy)

        return selected


class LeastConnectionsRouter:
    def choose(self, servers):
        healthy = [s for s in servers if s.available]

        if not healthy:
            raise RuntimeError("No healthy backend servers")

        return min(
            healthy,
            key=lambda server: (
                server.active_requests,
                server.queue_length,
            ),
        )


import joblib
import pandas as pd


class MLRouter:
    def __init__(self, model_path, fallback=None):
        self.model = joblib.load(model_path)
        self.fallback = fallback or LeastConnectionsRouter()

    def choose(self, servers, request_complexity=3):
        healthy = [s for s in servers if s.available]

        if not healthy:
            raise RuntimeError("No healthy backend servers")

        try:
            features = pd.DataFrame([
                {
                    "server_id": s.server_id,
                    "cpu_percent": s.cpu_percent,
                    "memory_percent": s.memory_percent,
                    "active_requests": s.active_requests,
                    "queue_length": s.queue_length,
                    "recent_avg_latency_ms": s.recent_avg_latency_ms,
                    "request_complexity": request_complexity,
                }
                for s in healthy
            ])

            predictions = self.model.predict(features)

            # Ignore invalid predictions rather than routing by them.
            candidates = [
                (server, float(latency))
                for server, latency in zip(healthy, predictions)
                if pd.notna(latency) and latency >= 0
            ]

            if not candidates:
                raise ValueError("Model produced no valid predictions")

            selected, predicted_latency = min(
                candidates,
                key=lambda item: item[1],
            )

            return selected

        except Exception:
            # Keep the service usable if model inference fails.
            return self.fallback.choose(healthy)
