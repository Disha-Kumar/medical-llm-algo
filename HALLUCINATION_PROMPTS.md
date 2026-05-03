# Hallucination Detection Prompt Templates

These templates take model output plus the ground-truth CheXpert label and flag
invented findings. They are intended for manual review or future LLM-judge use.

## Template 1: Strict Label-Level Judge

```text
You are reviewing a research-only chest X-ray model output.

Ground-truth CheXpert label:
{ground_truth}

Model output:
{model_output}

Task:
Decide whether the model invented or asserted a radiographic finding that is not
supported by the ground-truth label. Use the CheXpert label as the only source
of truth. Be conservative: if the output is empty, unparseable, or claims a
different disease than the ground truth, flag it.

Return exactly:
HALLUCINATION: yes/no
REASON: <one sentence>
UNSUPPORTED_FINDINGS: <comma-separated list or none>
```

## Template 2: Explanation-Focused Judge

```text
You are evaluating whether a medical vision-language model explanation contains
unsupported medical claims.

Ground truth:
{ground_truth}

Predicted diagnosis:
{diagnosis}

Model explanation:
{explanation}

Question:
Does the explanation introduce findings, causes, severity, laterality, devices,
or diseases that are not supported by the ground-truth label?

Return exactly:
HALLUCINATION: yes/no
REASON: <one sentence>
CONFIDENCE: <low/medium/high>
```

## Template 3: Structured JSON Judge

```text
You are a conservative label-level evaluator for CheXpert chest X-ray outputs.

Ground truth label:
{ground_truth}

Model diagnosis:
{diagnosis}

Model confidence:
{confidence}

Model explanation:
{explanation}

Flag hallucination if:
- the diagnosis is empty or unparseable,
- the diagnosis differs from the ground truth,
- the ground truth is "no finding" but the model asserts pathology,
- the explanation mentions findings not supported by the ground-truth label.

Return valid JSON only:
{
  "hallucination": true_or_false,
  "unsupported_findings": ["..."],
  "reason": "one sentence"
}
```

## Caveats

- These prompts are label-level screens, not radiologist adjudication.
- CheXpert labels are coarse and can omit secondary findings.
- A hallucination flag means "unsupported by available label," not necessarily
  impossible on the image.
- Empty or unparseable model output is flagged because it cannot support a
  diagnosis/explanation claim.

