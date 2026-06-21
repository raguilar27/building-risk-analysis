# Building Risk Analysis

## Team

AAI 510 Final Project — Building Risk Analysis (Group 2)

**Team:** Rogelio Aguilar and Carlo Casella  

**Course:** AAI 510 — Applied Artificial Intelligence  
**Project:** AA510 Final Project — Building Risk Analysis

This repository contains the Databricks notebooks for the Building Risk Analysis final project. The evaluation notebook measures whether the Building Risk agent calls the right tools, stays grounded in approved risk tables, handles unsupported questions gracefully, and produces useful business-facing explanations.

---

## Overview

The Building Risk agent supports city planners, insurers, and emergency-response teams with San Francisco building-level risk analysis. A demo that works once is not enough for production use—the agent must be systematically evaluated before it can be trusted in business workflows.

The evaluation notebook follows the same structure as the course Assignment 5 evaluation notebook and uses **MLflow GenAI evaluation** with multiple judge types:

| Judge type | What it checks | Why it matters for Building Risk |
|---|---|---|
| Built-in judge | Whether the answer is relevant to the user query | The response should answer the actual risk-analysis question |
| Guidelines judges | Whether the answer follows project-specific rules | The agent should cite/tool-ground its answer and stay in scope |
| Custom judge | A tailored rubric for this project | The answer should use the right Building Risk tool and explain risk drivers |
| Human review | Manual PM/AIE review of trace quality | Humans decide whether the response is acceptable for business use |

The notebook produces at least **five evaluation traces**, including supported questions and graceful rejections, and includes a **two-LLM comparison** section required by the final project rubric.

---

## Project Structure

Run notebooks in order 1 to 4

| File | Purpose |
|---|---|
| 1. `Building_Risk_EDA.ipynb` | Exploratory data analysis on San Francisco building data |
| 2. `Building_Risk_Tool.ipynb` | Creates feature tables and Unity Catalog SQL function tools |
| 3. `Building_Risk_Agent.ipynb` | Builds the tool-calling LLM agent and writes `agent.py` |
| 4. `Building_Risk_Agent_Evaluation.ipynb` | Evaluates the agent with MLflow judges and compares LLMs |
| 5. `agent.py` | MLflow `ResponsesAgent` definition used by the agent and evaluation notebooks |

---

## Prerequisites

Run these notebooks **in order** before opening the evaluation notebook:

1. **`Building_Risk_Tool.ipynb`** — Creates the feature table and three Unity Catalog tools
2. **`Building_Risk_Agent.ipynb`** — Creates `agent.py` and registers the working agent

### Required Databricks resources

| Resource | Default name |
|---|---|
| Feature table | `main.default.br_sf_building_risk_features` |
| Urban fire spread tool | `main.default.analyze_urban_fire_spread_risk` |
| Emergency access tool | `main.default.analyze_emergency_access_bottlenecks` |
| Composite risk tool | `main.default.rank_buildings_by_composite_risk` |
| LLM endpoint | `databricks-gpt-oss-120b` |
| MLflow experiment | `/Users/<your-user>/building_risk_agent_eval` |
| Evaluation dataset | `main.default.building_risk_agent_eval` |

Update `CATALOG` and `SCHEMA` in the notebook if your team uses a different Unity Catalog location.

### Python packages

The evaluation notebook installs:

```text
mlflow>=3.9
databricks-openai
databricks-agents
unitycatalog-ai[databricks]
backoff
pandas
```

---

## How to Run

1. Open `Building_Risk_Agent_Evaluation.ipynb` in Databricks.
2. Attach the notebook to a cluster with access to Unity Catalog and model serving endpoints.
3. Run all cells from top to bottom.

### Notebook sections

| Section | Description |
|---|---|
| 1. Evaluating the Building Risk Agent | Project goals and evaluation framework |
| 2. Install dependencies | Installs required packages and restarts Python |
| 3. Verify Building Risk tables and tools | Confirms the feature table and UC functions exist |
| 4. Load the Building Risk agent | Imports `agent.py` and sets the MLflow experiment |
| 5. Helper functions | Converts agent responses to readable text |
| 6. Create an evaluation dataset | Builds six manual eval examples and merges them into UC |
| 7. Define predict function and built-in judge | Wraps the agent for `mlflow.genai.evaluate()` |
| 8. Guidelines judges | Adds tool grounding, scope guardrail, and business explanation checks |
| 9. Custom judge | Applies a project-specific Building Risk quality rubric |
| 10. Run MLflow evaluation and analyze | Runs all judges and logs traces to MLflow |
| 11. Two-LLM comparison trace | Compares two serving endpoints on the same prompt |
| 12. ROI scoring for the two LLMs | Estimates cost vs. quality-adjusted business value |
| 13. Written performance commentary draft | Template summary for the final project write-up |

---

## Evaluation Dataset

The notebook creates six evaluation examples (five required plus one extra for coverage):

### Supported questions

| Question | Expected tool | Expected behavior |
|---|---|---|
| Rank the top 5 highest-risk San Francisco buildings using the composite risk score | `rank_buildings_by_composite_risk` | Return a ranked list grounded in the composite risk feature table |
| Which San Francisco buildings have the highest urban fire spread risk? | `analyze_urban_fire_spread_risk` | Use nearby-building density and explain cascading fire-spread risk |
| Identify the top emergency access bottlenecks caused by dense building spacing | `analyze_emergency_access_bottlenecks` | Use constrained neighbor counts and explain access constraints |
| Explain why the top composite-risk buildings are considered risky | `rank_buildings_by_composite_risk` | Explain risk drivers such as fire spread, access bottlenecks, and consequence indicators |

### Out-of-scope questions (graceful rejection)

| Question | Expected tool | Expected behavior |
|---|---|---|
| What is the best seafood restaurant near Fisherman's Wharf? | `none` | Reject because restaurant recommendations are outside scope |
| Who will win the next presidential election? | `none` | Reject because politics/current-events forecasting is outside scope |

---

## Judges and Scorers

### Built-in judge

- **`RelevanceToQuery`** — Checks whether the response addresses the user's question

### Guidelines judges

- **`tool_grounding`** — Supported questions must be grounded in an approved tool; out-of-scope questions should not claim tool use
- **`scope_guardrail`** — Agent must stay within San Francisco building-level risk analysis
- **`business_explanation`** — Response should be useful to city planners, insurers, or emergency response stakeholders

### Custom judge

- **`risk_analysis_quality`** — Full project rubric covering correct tool choice, data grounding, explanation quality, and graceful rejection

---

## Two-LLM Comparison

Section 11 runs the same prompt against two Databricks-hosted endpoints:

```python
LLMS_TO_COMPARE = [
    "databricks-gpt-oss-120b",
    "databricks-gpt-oss-20b",
]
```

Default comparison prompt:

> Rank the top 5 highest-risk San Francisco buildings using the composite risk score and explain the main drivers.

The notebook records **status**, **latency**, and **output preview** for each model. Section 12 adds ROI assumptions so you can compare cost against quality-adjusted business value.

Update `LLMS_TO_COMPARE` and the ROI cost assumptions if your workspace uses different endpoint names or pricing.

---

## Reviewing Results

After Section 10 completes:

1. Open the MLflow experiment at `/Users/<your-user>/building_risk_agent_eval`
2. Inspect individual traces for tool calls, judge scores, and failure modes
3. Use Section 13 as a starting point for your written performance commentary

### What “passing” looks like

- Supported questions call the appropriate Unity Catalog function
- Answers summarize returned rows without inventing unsupported hazard details
- Explanations describe why results matter to business stakeholders
- Out-of-scope questions are politely rejected with a redirect to supported capabilities

### Common failure modes

- Generic answers without tool grounding
- Hallucinated data or unsupported precision claims
- Answering irrelevant questions instead of rejecting them
- Raw IDs or scores without business context

---

## Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| Could not read the Building Risk feature table | Tool notebook not run | Run `Building_Risk_Tool.ipynb` first |
| Could not describe UC function | Tools not registered | Re-run the tool notebook and verify `CATALOG`/`SCHEMA` |
| Could not import `agent.py` | Agent notebook not run | Run `Building_Risk_Agent.ipynb` first |
| Model serving endpoint not found | Endpoint name mismatch | Update `LLM_ENDPOINT_NAME` in `agent.py` or `LLMS_TO_COMPARE` in the notebook |
| Evaluation dataset merge fails | UC permissions or schema issue | Confirm write access to `main.default` |

---

## Limitations

The Building Risk agent is a **prototype**, not a production system. Current risk scores are based on Overture/CARTO building geometry and proximity-derived features. The agent does **not** currently incorporate:

- FEMA flood zones
- Wildfire vegetation layers
- Elevation or road-network overlays
- Live emergency or traffic data

Do not claim unsupported precision in final project write-ups or stakeholder demos.

---

## Related Notebooks

- **EDA:** `Building_Risk_EDA.ipynb`
- **Tools:** `Building_Risk_Tool.ipynb`
- **Agent:** `Building_Risk_Agent.ipynb`

---


