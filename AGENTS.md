# AGENTS.md

## Project context

- This repository is based on a fork of an existing research codebase.
- The goal is to extend the original work while preserving the ability to reproduce the upstream baseline.
- Treat upstream behavior as a reference implementation unless the task explicitly requires changing it.
- Prefer additive extensions over modifying original logic in place.
- Use `.venv` for python environment.

## PROJECT_CONTEXT rules

- This repository should maintain a `PROJECT_CONTEXT/` directory as the primary project memory for context, status, plans, and key decisions.
- Before making substantial changes, read the relevant files in `PROJECT_CONTEXT/` to understand the current project direction, constraints, and planned work.
- Treat `PROJECT_CONTEXT/` as a working reference together with the codebase, configs, and existing docs.
- If the repository state and `PROJECT_CONTEXT/` disagree, identify the mismatch and update the context files when appropriate.
- When implementation, plans, experiment direction, assumptions, or project status change meaningfully, update the relevant files in `PROJECT_CONTEXT/`.
- Prefer updating an existing context file instead of creating a redundant new one.
- Keep `PROJECT_CONTEXT/` organized into focused markdown files with clear purposes.
- Prefer keeping `PROJECT_CONTEXT/` to 10 or fewer markdown files whenever practical.
- Prefer keeping each markdown file within about 300 lines whenever practical.
- Keep entries concise, structured, and directly useful for future work.
- Record important changes such as:
  - current project status
  - active plan or next steps
  - architecture or implementation decisions
  - experiment assumptions
  - known risks, blockers, or open questions
- Do not store raw logs, long transcripts, or low-signal notes in `PROJECT_CONTEXT/`.
- When code changes affect project understanding, make sure the corresponding `PROJECT_CONTEXT/` files are updated to stay consistent with the repository.

## Fork-first principles

- Preserve the original repository behavior whenever possible.
- Minimize intrusive edits to upstream code.
- New ideas, modules, and experiment paths should be added in a clearly separated manner.
- Keep it easy to identify what belongs to upstream and what belongs to the fork extension.
- Do not silently rewrite original baseline logic.

## Upstream compatibility

- Do not rename or restructure core upstream files unless necessary.
- Do not break original training, evaluation, or inference commands without explicit reason.
- If upstream scripts must be changed, keep the change minimal and backward-compatible where practical.
- Prefer wrappers, adapters, subclassing, new config files, or new entrypoints over destructive rewrites.
- Keep original experiment paths reproducible.

## Extension strategy

- Add new methods in clearly named modules, files, configs, or directories.
- Use naming that distinguishes fork-specific additions from upstream components.
- Keep baseline, fork extension, and ablation logic clearly separable.
- Prefer feature flags, config switches, or separate experiment configs for new behavior.
- Ensure new methods can be enabled or disabled cleanly.

## Editing rules

- Read the upstream structure and current repository flow before editing.
- Follow the existing code style and repository conventions unless there is a strong reason not to.
- Keep edits minimal, local, and easy to review.
- Do not hardcode local paths, credentials, or machine-specific settings.
- Prefer config-driven behavior over inline constants.
- Avoid broad refactors unless they are required for the extension.

## Experiment integrity

- Do not silently change default hyperparameters, preprocessing, augmentation, dataset split logic, loss definitions, postprocessing, or evaluation metrics in upstream baselines.
- If baseline-related behavior must change, clearly isolate and document the change.
- Keep original baseline configs intact whenever possible.
- Add separate configs for fork-specific experiments rather than mutating upstream defaults.
- Do not overwrite existing checkpoints, logs, or outputs unless explicitly instructed.

## Reproducibility

- Preserve the ability to run the original method as close to upstream behavior as possible.
- Keep random seed handling explicit.
- Keep output directories structured so baseline and extension results are not mixed.
- Record assumptions, deviations from upstream, and fork-specific settings in markdown or config comments.
- Clearly distinguish reproduction settings from newly proposed settings.

## Validation expectations

- After code changes, run the smallest meaningful validation first.
- Prefer the following order when feasible:
  1. syntax / import check
  2. module-level or unit test
  3. small forward-pass or smoke test
  4. targeted experiment check
- If the original baseline flow may have been affected, validate that flow explicitly when feasible.
- Never claim baseline reproduction or experimental gains were verified unless actually executed.

## Documentation rules

- Document fork-specific changes clearly.
- Keep a short summary of:
  - what remains identical to upstream
  - what was added in the fork
  - what was modified from upstream
- Prefer updating existing docs or adding concise fork notes rather than writing redundant long documents.
- Add short usage notes for new configs, new modules, and new experiment entrypoints.

## Communication rules

- Before substantial edits, briefly state whether the change affects:
  1. upstream baseline path
  2. fork extension path
  3. both
- When finishing, report:
  1. files changed
  2. whether each change was upstream-touching or fork-specific
  3. what was implemented
  4. what was tested
  5. remaining risks or unverified points
- Clearly distinguish repository facts, assumptions, and unverified expectations.

## Git workflow

- Keep commits separated by baseline-touching changes and fork-specific feature changes whenever practical.

## Things to avoid

- Do not replace original baseline code with fork-specific experimental code unless explicitly requested.
- Do not mix baseline edits and novel-method edits in a way that makes comparison unclear.
- Do not fabricate reproduction success, benchmark numbers, or validation outcomes.
- Do not make aesthetic-only refactors across upstream files without need.
- Do not remove upstream options, configs, or execution paths unless explicitly requested.

## Project-specific references

- Use `PROJECT_CONTEXT/references` for actual papers
