import asyncio
import csv
import random
import time
from collections import deque
from pathlib import Path


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "load_balancer_dataset.csv"

# Experiment configuration
TOTAL_REQUESTS = 3000
BATCH_SIZE = 30

FEATURE_COLUMNS = [
    "server_id",
    "cpu_percent",
    "memory_percent",
    "active_requests",
    "queue_length",
    "recent_avg_latency_ms",
    "request_complexity",
    "response_time_ms",
]


class BackendServer:
    def __init__(self, server_id, capacity, base_latency_ms):
        self.server_id = server_id
        self.capacity = capacity
        self.base_latency_ms = base_latency_ms

        # Limit the number of requests processed simultaneously.
        self.semaphore = asyncio.Semaphore(capacity)

        self.active_requests = 0
        self.queued_requests = 0
        self.lock = asyncio.Lock()

        # Keep recent response times for a rolling average.
        self.recent_latencies = deque(maxlen=50)

    async def handle_request(self, request_complexity):
        # Capture the state before this request joins the queue.
        async with self.lock:
            active_before = self.active_requests
            queue_before = self.queued_requests

            background_load = random.uniform(0, 12)

            cpu_percent = min(
                98.0,
                15
                + active_before * 12
                + queue_before * 3
                + background_load,
            )

            memory_percent = min(
                95.0,
                30
                + active_before * 5
                + queue_before * 2
                + random.uniform(0, 8),
            )

            if self.recent_latencies:
                recent_avg_latency_ms = (
                    sum(self.recent_latencies)
                    / len(self.recent_latencies)
                )
            else:
                recent_avg_latency_ms = self.base_latency_ms

            self.queued_requests += 1

        # Start measuring actual elapsed request time, including waiting.
        start_time = time.perf_counter()

        try:
            async with self.semaphore:
                async with self.lock:
                    self.queued_requests -= 1
                    self.active_requests += 1

                try:
                    # Simulate backend processing time.
                    processing_delay_ms = (
                        self.base_latency_ms
                        + request_complexity * 18
                        + cpu_percent * 0.6
                        + active_before * 8
                        + random.uniform(0, 20)
                    )

                    await asyncio.sleep(
                        processing_delay_ms / 1000
                    )
                finally:
                    async with self.lock:
                        self.active_requests -= 1

        finally:
            # Measure the total time, including queue waiting.
            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

        async with self.lock:
            self.recent_latencies.append(elapsed_ms)

        return {
            "server_id": self.server_id,
            "cpu_percent": round(cpu_percent, 2),
            "memory_percent": round(memory_percent, 2),
            "active_requests": active_before,
            "queue_length": queue_before,
            "recent_avg_latency_ms": round(
                recent_avg_latency_ms, 2
            ),
            "request_complexity": request_complexity,
            "response_time_ms": round(elapsed_ms, 2),
        }


async def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Different capacities and speeds create heterogeneous servers.
    servers = [
        BackendServer("server_1", capacity=3, base_latency_ms=35),
        BackendServer("server_2", capacity=5, base_latency_ms=50),
        BackendServer("server_3", capacity=2, base_latency_ms=25),
    ]

    rows = []

    for batch_start in range(0, TOTAL_REQUESTS, BATCH_SIZE):
        current_batch_size = min(
            BATCH_SIZE,
            TOTAL_REQUESTS - batch_start,
        )

        tasks = []

        for i in range(current_batch_size):
            # Distribute experiments evenly across the three servers.
            server_index = (batch_start + i) % len(servers)
            server = servers[server_index]

            # 1 = light request, 5 = heavy request.
            complexity = random.randint(1, 5)

            tasks.append(
                server.handle_request(complexity)
            )

        batch_results = await asyncio.gather(*tasks)
        rows.extend(batch_results)

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FEATURE_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Collection complete: {len(rows)} rows")
    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())