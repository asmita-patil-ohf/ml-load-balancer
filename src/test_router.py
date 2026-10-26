
from pathlib import Path

from router import (
    BackendServer,
    RoundRobinRouter,
    LeastConnectionsRouter,
    MLRouter,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "latency_model.joblib"


def create_servers():
    return [
        BackendServer(
            server_id="server_1",
            url="http://localhost:8001",
            cpu_percent=35,
            memory_percent=45,
            active_requests=2,
            queue_length=0,
            recent_avg_latency_ms=65,
        ),
        BackendServer(
            server_id="server_2",
            url="http://localhost:8002",
            cpu_percent=85,
            memory_percent=75,
            active_requests=8,
            queue_length=4,
            recent_avg_latency_ms=180,
        ),
        BackendServer(
            server_id="server_3",
            url="http://localhost:8003",
            cpu_percent=55,
            memory_percent=50,
            active_requests=3,
            queue_length=1,
            recent_avg_latency_ms=90,
        ),
    ]


def main():
    servers = create_servers()

    round_robin = RoundRobinRouter()
    least_connections = LeastConnectionsRouter()
    ml_router = MLRouter(MODEL_PATH)

    print("--- Round Robin ---")
    for _ in range(6):
        print(round_robin.choose(servers).server_id)

    print("\n--- Least Connections ---")
    print(least_connections.choose(servers).server_id)

    print("\n--- ML-Based Routing ---")
    print(ml_router.choose(servers, request_complexity=3).server_id)

    print("\n--- Unhealthy server test ---")
    servers[0].is_healthy = False

    selected = ml_router.choose(
        servers,
        request_complexity=3,
    )
    print("Selected:", selected.server_id)

    assert selected.server_id != "server_1"

    print("\nAll routing smoke tests passed.")


if __name__ == "__main__":
    main()
