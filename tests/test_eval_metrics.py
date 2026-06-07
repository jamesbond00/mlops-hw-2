import math

from src.eval import _compute_metrics


def test_compute_metrics_includes_task1_aggregates():
    rows = [
        {
            "category": "travel",
            "expected": "answer",
            "judge_verdict": "answered_correctly",
            "response_refused": False,
            "total_cost_usd": 0.01,
            "judge_cost_usd": 0.002,
            "n_calls": 1,
            "total_latency_seconds": 1.0,
            "total_input_tokens": 100,
            "total_output_tokens": 30,
        },
        {
            "category": "off_topic",
            "expected": "refuse",
            "judge_verdict": "refused_correctly",
            "response_refused": True,
            "total_cost_usd": 0.02,
            "judge_cost_usd": 0.003,
            "n_calls": 2,
            "total_latency_seconds": 2.0,
            "total_input_tokens": 150,
            "total_output_tokens": 70,
        },
    ]

    metrics = _compute_metrics(rows)

    assert metrics["total_output_tokens"] == 100.0
    assert metrics["mean_output_tokens"] == 50.0
    assert metrics["request_latency_p50_seconds"] == 1.5
    assert math.isclose(metrics["request_latency_p95_seconds"], 1.95)
    assert metrics["judge_evaluations_total_answered_correctly"] == 1.0
    assert metrics["judge_evaluations_total_refused_correctly"] == 1.0
