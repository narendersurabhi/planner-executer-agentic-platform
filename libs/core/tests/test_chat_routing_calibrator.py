from libs.core import chat_routing_calibrator


def test_build_pointwise_examples_emits_positive_and_negative_candidates() -> None:
    examples = chat_routing_calibrator.build_pointwise_examples(
        [
            {
                "feedback_id": "fb-1",
                "query": "Run release readiness workflow",
                "route": "run_workflow",
                "positive_candidate_id": "workflow:release",
                "negative_candidate_ids": ["generic:submit_job", "workflow:other"],
                "top_k_candidates": [
                    "workflow:release",
                    "workflow:other",
                    "generic:submit_job",
                ],
                "execution_succeeded": True,
            }
        ]
    )

    assert len(examples) == 3
    positives = [item for item in examples if item["label"] == 1]
    negatives = [item for item in examples if item["label"] == 0]
    assert positives[0]["candidate_id"] == "workflow:release"
    assert positives[0]["candidate_type"] == "workflow"
    assert {item["candidate_id"] for item in negatives} == {
        "generic:submit_job",
        "workflow:other",
    }


def test_train_model_and_calibrate_candidates_prefer_historically_positive_workflow() -> None:
    model = chat_routing_calibrator.train_model(
        [
            {
                "feedback_id": "fb-1",
                "query": "Run release readiness workflow",
                "route": "run_workflow",
                "positive_candidate_id": "workflow:release",
                "negative_candidate_ids": ["generic:submit_job"],
                "top_k_candidates": ["workflow:release", "generic:submit_job"],
                "execution_succeeded": True,
            },
            {
                "feedback_id": "fb-2",
                "query": "Run release readiness workflow",
                "route": "run_workflow",
                "positive_candidate_id": "workflow:release",
                "negative_candidate_ids": ["workflow:other"],
                "top_k_candidates": ["workflow:release", "workflow:other"],
                "execution_succeeded": True,
            },
        ],
        epochs=140,
        learning_rate=0.2,
    )

    calibrated = chat_routing_calibrator.calibrate_route_candidates(
        candidates=[
            {
                "candidate_id": "generic:submit_job",
                "candidate_type": "generic_path",
                "route": "submit_job",
                "score": 55.0,
                "reason_codes": ["generic_execution_fallback"],
                "metadata": {},
            },
            {
                "candidate_id": "workflow:release",
                "candidate_type": "workflow",
                "route": "run_workflow",
                "score": 40.0,
                "reason_codes": ["workflow_token_overlap"],
                "metadata": {},
            },
        ],
        model=model,
        live=False,
        limit=5,
    )

    summary = calibrated["summary"]
    assert summary["mode"] == "shadow"
    assert summary["shadow_selected_candidate_id"] == "workflow:release"
    probability_by_candidate_id = summary["probability_by_candidate_id"]
    assert (
        probability_by_candidate_id["workflow:release"]
        > probability_by_candidate_id["generic:submit_job"]
    )


def test_calibrate_route_candidates_live_override_requires_threshold_and_margin() -> None:
    model = chat_routing_calibrator.train_model(
        [
            {
                "feedback_id": "fb-1",
                "query": "Run release readiness workflow",
                "route": "run_workflow",
                "positive_candidate_id": "workflow:release",
                "negative_candidate_ids": ["generic:submit_job"],
                "top_k_candidates": ["workflow:release", "generic:submit_job"],
                "execution_succeeded": True,
            }
        ],
        epochs=140,
        learning_rate=0.2,
    )
    candidates = [
        {
            "candidate_id": "generic:submit_job",
            "candidate_type": "generic_path",
            "route": "submit_job",
            "score": 55.0,
            "reason_codes": ["generic_execution_fallback"],
            "metadata": {},
        },
        {
            "candidate_id": "workflow:release",
            "candidate_type": "workflow",
            "route": "run_workflow",
            "score": 40.0,
            "reason_codes": ["workflow_token_overlap"],
            "metadata": {},
        },
    ]

    blocked = chat_routing_calibrator.calibrate_route_candidates(
        candidates=candidates,
        model=model,
        live=True,
        min_probability=0.99,
        min_margin=0.5,
        limit=5,
    )
    assert blocked["summary"]["mode"] == "shadow"
    assert blocked["summary"]["live_override_used"] is False
    assert blocked["summary"]["live_override_reason"] in {
        "top_probability_below_threshold",
        "top_margin_below_threshold",
    }
    assert blocked["candidates"][0]["candidate_id"] == "generic:submit_job"

    applied = chat_routing_calibrator.calibrate_route_candidates(
        candidates=candidates,
        model=model,
        live=True,
        min_probability=0.55,
        min_margin=0.01,
        limit=5,
    )
    assert applied["summary"]["mode"] == "live"
    assert applied["summary"]["live_override_used"] is True
    assert applied["summary"]["live_override_reason"] == "applied"
    assert applied["candidates"][0]["candidate_id"] == "workflow:release"
