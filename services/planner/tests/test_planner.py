from datetime import UTC, datetime

from libs.core import models
from services.planner.app.main import _parse_llm_plan, rule_based_plan


def test_rule_based_plan_schema():
    job = models.Job(
        id="job",
        goal="demo",
        context_json={},
        status=models.JobStatus.queued,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        priority=1,
        metadata={},
    )
    plan = rule_based_plan(job, [])
    assert plan.planner_version
    assert len(plan.tasks) == 3


def test_parse_llm_plan_accepts_fenced_plan_with_null_schema_ref():
    plan = _parse_llm_plan(
        """```json
{
  "planner_version": "1.0.0",
  "tasks_summary": "Generate and render a DOCX.",
  "dag_edges": [["GenerateDocumentSpec", "RenderDocxDocument"]],
  "tasks": [
    {
      "name": "GenerateDocumentSpec",
      "description": "Generate a document spec.",
      "instruction": "Generate a document spec.",
      "acceptance_criteria": ["Spec produced"],
      "expected_output_schema_ref": "schemas/document_spec",
      "intent": "generate",
      "deps": [],
      "capability_requests": ["document.spec.generate"],
      "tool_inputs": {
        "document.spec.generate": {
          "instruction": "Generate a document spec."
        }
      },
      "critic_required": true
    },
    {
      "name": "RenderDocxDocument",
      "description": "Render the document spec.",
      "instruction": "Render the document spec to DOCX.",
      "acceptance_criteria": ["DOCX produced"],
      "expected_output_schema_ref": null,
      "intent": "render",
      "deps": ["GenerateDocumentSpec"],
      "capability_requests": ["document.docx.render"],
      "tool_inputs": {
        "document.docx.render": {
          "document_spec": {
            "$from": "dependencies_by_name.GenerateDocumentSpec.output"
          },
          "path": "agenticaiops.docx"
        }
      },
      "critic_required": false
    }
  ]
}
```"""
    )

    assert plan is not None
    assert plan.tasks[1].expected_output_schema_ref == "schemas/docx_output"
