# Guarded LLM Orchestrator: Validated Decision Routing with Phi-3.5

An experimental **Agentic AI workflow** that sends a free-text request to
a small language model, forces it to answer with a structured JSON
decision, validates that decision against a fixed set of allowed
actions, and attempts one automatic repair pass when the output is
invalid.

The notebook demonstrates a layered approach to building a reliable
orchestrator around a locally hosted small language model
(**Phi-3.5-mini-instruct**), with an emphasis on 4-bit quantized
loading, strict output contracts, graceful recovery from imperfect model
output, and clear separation between model, orchestration, and UI
layers.

> **Project:** Agentic Workflow (Guarded Orchestrator)\
> **Notebook:** `Agentic_workflow.ipynb`\
> **Domain:** Agentic AI · LLMs · Python · Structured Output · Output
> Validation · Local Inference\
> **Environment:** Google Colab (T4 GPU)

------------------------------------------------------------------------

## 1. Project Overview

The goal is to let a user enter a task or request and receive a
**machine-readable decision** from an LLM orchestrator, where the
decision is guaranteed by application code to belong to a known set of
actions.

Instead of trusting the model's raw text, the workflow places
programmatic controls between generation and the final result:

1.  Load Phi-3.5-mini-instruct locally with 4-bit quantization.
2.  Instruct the model, through a system prompt, to answer only with a
    JSON block containing `decision`, `role`, and `reason`.
3.  Generate a response with low-temperature sampling.
4.  Extract the JSON object from the raw text.
5.  Validate that the `decision` field exists and is exactly one of the
    allowed actions.
6.  If validation fails, run a dedicated repair pass that shows the
    model its faulty output and asks for a corrected one.
7.  Render the outcome, including the raw model logs, in an interactive
    notebook UI.

The allowed actions in the notebook are:

-   `Create Ticket`
-   `Ask for more info`
-   `Reject the request`

The central engineering principle is:

> **A prompt is not a guarantee. The application code must validate
> every model output before it is accepted as a decision.**

------------------------------------------------------------------------

## 2. Key Features

### Local, memory-efficient model loading

-   Loads `microsoft/Phi-3.5-mini-instruct` through Hugging Face
    Transformers.
-   Uses **4-bit NF4 quantization** with double quantization and
    `float16` compute via `bitsandbytes`.
-   Enables FP32 CPU offload and `device_map="auto"` so the model can
    fit on a limited GPU such as the Colab T4.
-   Sets `expandable_segments` in the CUDA allocator configuration to
    reduce memory fragmentation.

### Structured output contract

-   The system prompt requires a JSON-only response.
-   The `decision` field is constrained to an explicit allow-list that
    is injected into the prompt from the same variable used for
    validation, so the two cannot drift apart.
-   The response also includes `role` and `reason` fields for
    explainability.

### Output extraction and validation

-   Extracts the JSON object from surrounding text with a regular
    expression and parses it with `json.loads`.
-   Returns `None` on any parsing failure instead of raising.
-   Validates both the presence of the `decision` key and the membership
    of its value in the allowed actions.
-   Returns a human-readable validation message for every outcome.

### Repair pass

-   Isolated in its own method, `repair_response`.
-   Appends the faulty output and a corrective user message to a copy of
    the original conversation, then regenerates.
-   Re-extracts and re-validates the new output.
-   Labels the validation message with `[Repair Pass]` so the origin of
    the final result is visible.
-   Runs a **single** repair attempt; there is no unbounded retry loop.

### Resource management

-   `setup_service(force=True)` deletes the active model service, runs
    garbage collection, and clears the CUDA cache.
-   Re-uses an already loaded model by default to avoid repeated loading
    in an interactive session.
-   Includes commented recovery steps for out-of-memory situations.

### Interactive interface

-   Built with `ipywidgets`: a task text area, an execute button, and a
    bordered output area.
-   Displays the validation status, the parsed decision fields, and a
    collapsible accordion with the raw model logs.
-   Renders failures with any partially extracted data for debugging.

------------------------------------------------------------------------

## 3. Architecture

| Component | Responsibility |
|-----------|----------------|
| `PhiModelService` | Loads the tokenizer and quantized model, builds the generation pipeline, and produces raw text from chat messages. |
| `setup_service` | Creates or re-uses the model service and supports a forced reset with GPU memory cleanup. |
| `PhiOrchestrator` | Builds the guarded prompt, parses the response, validates the decision, and coordinates the repair pass. |
| `extract_json` | Finds and parses the JSON object in raw model text. |
| `validate_decision` | Checks the `decision` key and verifies the value against the allowed actions. |
| `repair_response` | Re-prompts the model with its faulty output and re-validates the result. |
| `ask_llm_guarded` | Entry point that runs generation, validation, and conditional repair. |
| `ResultRenderer` | Formats success and failure outcomes and the raw log accordion. |
| `OrchestratorUI` | Manages widgets, layout, and the button-click workflow. |

### High-level workflow

``` text
        User task (text input)
                  |
                  v
      Build guarded messages
   (system prompt + allowed actions)
                  |
                  v
          Language model
        (Phi-3.5-mini, 4-bit)
                  |
                  v
        Extract JSON from output
                  |
                  v
        Validate 'decision' field
           /              \
        Invalid          Valid
           |               |
           v               v
     Repair pass      Return result
  (one attempt only)  + raw logs
           |
           v
     Extract + validate again
           |
           v
     Return result + raw logs
     (valid or marked invalid)
```

------------------------------------------------------------------------

## 4. Technical Highlights

### Allow-list validation

The decision is accepted only if it exactly matches one of the allowed
actions. The same list is used to construct the system prompt and to
validate the output.

This is deny-by-default behavior: anything that is not an exact member
of the allow-list, including a missing key, a misspelled action, or a
different casing, is rejected.

### Separation of concerns

The notebook splits responsibilities into three layers:

-   **Service layer:** model loading, quantization, and raw generation.
-   **Orchestration layer:** prompting, parsing, validation, and repair.
-   **Presentation layer:** rendering and widget interaction.

The orchestrator depends on the service through its `generate` method
only, which makes it straightforward to swap the model or test the
orchestration logic with a stub.

### Conversation-aware repair

The repair pass does not simply retry. It extends the original
conversation with the model's faulty answer and an explicit correction
request, so the model can see what it produced and what was wrong with
it. The original message list is copied before modification, so the
first-pass conversation is not mutated.

### Low-temperature sampling

Generation uses `temperature=0.1` with `do_sample=True` and
`max_new_tokens=500`. The low temperature keeps outputs close to
deterministic while preserving the sampling configuration required by
the pipeline. It does not guarantee identical outputs across runs.

### Transparent logging

Every result is returned together with the raw model output, and the
UI exposes it through an accordion. This makes it easy to see why a
response was accepted, repaired, or rejected.

------------------------------------------------------------------------

## 5. Technology Stack

| Technology | Role |
|------------|------|
| Python | Core implementation language. |
| Google Colab (T4 GPU) | Notebook execution environment. |
| PyTorch | Model inference backend and CUDA memory management. |
| Hugging Face Transformers | Model and tokenizer loading; text-generation pipeline; chat templating. |
| Accelerate | Device placement through `device_map="auto"`. |
| bitsandbytes | 4-bit NF4 quantization. |
| Phi-3.5-mini-instruct | Small language model used for decisions. |
| `json` / `re` | Response extraction and parsing. |
| `gc` / `os` | Memory cleanup and environment configuration. |
| `ipywidgets` / `IPython.display` | Interactive notebook interface. |

------------------------------------------------------------------------

## 6. Running the Notebook

### Requirements

The notebook is designed for Google Colab and a CUDA-capable GPU. The
4-bit quantized model is intended to fit on a free-tier T4.

Before running:

-   Open the notebook in Google Colab.
-   Select a **GPU runtime** (for example, T4).
-   Allow the first run to download the model weights from Hugging Face.
-   Run the setup cell to install `transformers`, `accelerate`, and
    `bitsandbytes`.

`torch` and `ipywidgets` are normally preinstalled in Colab. If you run
the notebook elsewhere, install them separately.

### Execution order

Run the notebook cells from top to bottom. The main stages are:

1.  Dependency installation.
2.  Model service definition and initialization (with OOM cleanup
    helper).
3.  Orchestrator definition (prompting, parsing, validation, repair).
4.  UI and renderer definition, followed by rendering the interface.

### Example interaction

Enter a request in the **Task** text area, for example:

``` text
Initiate system diagnostic
```

(this is the notebook's default input), then press **Execute
Orchestration**.

The workflow generates a JSON decision, validates it, optionally
repairs it, and displays the validation status, the parsed fields, and
the raw model logs. The exact decision depends on the model's output
for the given input.

------------------------------------------------------------------------

## 7. Configuration

The notebook keeps its main settings in a few places so they can be
adjusted without changing the orchestration logic.

Review these settings:

-   **Model identifier:** the `model_id` argument of `PhiModelService`.
-   **Quantization:** 4-bit, NF4, double quantization, `float16`
    compute, and CPU offload.
-   **Generation:** `max_tokens` (default 500), `temperature` (0.1), and
    sampling mode.
-   **Allowed actions:** the `allowed_actions` list in
    `PhiOrchestrator`.
-   **System prompt:** the instruction text in `ask_llm_guarded`.
-   **Repair message:** the correction text in `repair_response`.

If you change the set of allowed actions, update only the
`allowed_actions` list. The prompt and the validator both read from it.

Do not commit access tokens or credentials to a public repository. If a
gated model or private repository is used, store the Hugging Face token
in Colab secrets or an environment variable.

------------------------------------------------------------------------

## 8. Security Considerations and Limitations

This notebook is an educational prototype, not a production-grade
decision service.

| Area | Current approach | Important limitation |
|------|------------------|----------------------|
| Prompt injection | System prompt defines the output contract. | The user message goes directly to the model; a prompt cannot enforce behavior, and malicious input may steer the decision. |
| Validation depth | Checks that `decision` exists and is allowed. | `role` and `reason` are not validated for presence, type, or content. |
| JSON extraction | Regex capture of the outermost braces, then `json.loads`. | Greedy matching can capture invalid spans if the output contains multiple objects or extra braces. |
| Repair | A single repair pass. | A repaired response can still be invalid; the final result is then marked invalid and not acted upon. |
| Determinism | `temperature=0.1`. | Sampling remains enabled, so identical inputs may produce different decisions. |
| Decision quality | Allow-list membership. | A valid action is not necessarily the correct action; semantic correctness is not verified. |
| Resources | Quantization and cleanup helper. | No input-length limit, timeout, or rate limiting is applied. |
| Observability | Raw logs shown in the UI. | Logs are displayed only; they are not persisted or tamper-proof. |

### Recommendations before production use

-   Validate the full response schema, including `role` and `reason`.
-   Replace regex extraction with stricter parsing or constrained
    decoding.
-   Add input length limits, timeouts, and rate limiting.
-   Treat all user text as untrusted and test against prompt-injection
    attempts.
-   Require human review or additional checks before acting on
    consequential decisions.
-   Persist decisions and failures with timestamps and session
    metadata.
-   Evaluate decision accuracy on a labeled test set before relying on
    the model.

------------------------------------------------------------------------

## 9. Repository Structure

``` text
.
├── Agentic_workflow.ipynb
└── README.md
```

Model weights are downloaded at runtime and are not stored in the
repository.

------------------------------------------------------------------------

## 10. Possible Future Improvements

-   Validate the complete JSON schema (types and required fields).
-   Use constrained or grammar-guided decoding to guarantee valid JSON.
-   Allow a configurable number of repair attempts with a clear
    stopping condition.
-   Add unit tests for `extract_json`, `validate_decision`, and the
    repair flow using a stubbed model service.
-   Build a labeled evaluation set and measure decision accuracy,
    repair rate, and latency.
-   Add few-shot examples to the prompt for each allowed action.
-   Support Arabic and English inputs explicitly and evaluate both.
-   Log decisions to a persistent store with timestamps.
-   Compare Phi-3.5-mini with other small and larger models under the
    same evaluation protocol.

------------------------------------------------------------------------

## 11. Key Takeaway

This project demonstrates how an LLM can be integrated into a guarded
decision workflow: the model proposes a structured decision, while the
surrounding application is responsible for extraction, validation,
recovery, and visibility into what actually happened.

The most important design lesson is that **reliable agentic systems
require more than a capable model or a carefully written prompt: they
need explicit application-level checks on every output the model
produces.**

------------------------------------------------------------------------

## Author

Add your name, GitHub profile, and portfolio link here.

## License

No license has been specified. Add a `LICENSE` file before presenting
the repository as an open-source project.
