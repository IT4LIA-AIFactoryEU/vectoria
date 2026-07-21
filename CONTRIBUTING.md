# Contributing to Vectoria

Thank you for considering a contribution to Vectoria. This project provides
on-premise retrieval-augmented generation for private document question
answering.

## Before You Start

Please review the [README](README.md) for installation, configuration, and
development commands. For security-sensitive findings, follow the private
process in [SECURITY.md](SECURITY.md), not the public issue tracker.

## Reporting Bugs

Report bugs in the [GitLab issue tracker](https://gitlab.hpc.cineca.it/mceccar2/vectoria/-/issues).
Before opening an issue, search existing reports to avoid duplicates.

Include the following information:

- a clear, descriptive title and a concise description of the observed result;
- steps to reproduce the issue, including a minimal example where possible;
- the expected result and the actual result;
- the Vectoria revision or version, operating system, and deployment method;
- relevant logs, stack traces, and configuration details with secrets removed.

## Requesting Features

Request new functionality through the [GitLab issue tracker](https://gitlab.hpc.cineca.it/mceccar2/vectoria/-/issues).
Describe the problem the feature would solve, the proposed behaviour, any
alternatives considered, and relevant deployment or performance constraints.
Please discuss substantial changes in an issue before investing in an
implementation.

## Contribution Workflow

1. Open or identify an issue for the change, unless it is a small documentation
   correction.
2. Fork the repository and create a focused branch from the current default
   branch.
3. Make the smallest coherent change and update documentation and tests when
   applicable.
4. Run the relevant checks locally.
5. Open a merge request against the repository's default branch. Link the
   related issue, explain the change, and describe the checks performed.
6. Address review feedback and keep the merge request focused on its stated
   purpose.

Maintainers review contributions based on correctness, security, tests,
documentation, compatibility, and fit with the project scope. The merge
request is merged only after the required review and checks are complete.

## Development Checks

Set up the development environment and run the checks documented in the README:

```bash
uv sync --extra api --extra docling --extra observability
uv run pytest
uv run pytest -m "not slow"
uv run ruff check . && uv run ruff format .
```

Run the checks relevant to your change before opening a merge request. Explain
any check that cannot be run and why in the merge request description.

## Code and Writing Style

- Follow the existing structure, naming, and patterns in the code you modify.
- Use Python code that passes Ruff checks and is formatted with `ruff format`.
- Add or update focused tests for behavioural changes and bug fixes.
- Keep changes small, readable, and limited to the stated purpose.
- Write documentation, issue reports, and merge-request descriptions in clear
  English.
- Do not commit generated artifacts, credentials, API keys, private documents,
  or environment-specific configuration.

## Questions and Support

For general support or access problems related to the IT4LIA GitHub/GitLab
organization, contact [support-it4lia@cineca.it](mailto:support-it4lia@cineca.it).
