# Issue management

## Purpose

Issues are the project's traceable channel for reporting defects, proposing improvements, requesting clarification, and coordinating contributions. Before opening a new issue, contributors should search both open and closed issues for an equivalent report or request.

## Label taxonomy

| Label | Intended use |
|---|---|
| `bug` | Unexpected or incorrect behavior. |
| `documentation` | Corrections, additions, or improvements to documentation. |
| `enhancement` | A new feature or improvement to existing behavior. |
| `question` | A request for clarification, information, or support. |
| `good first issue` | A small, well-scoped task suitable for a new contributor. |
| `help wanted` | A task for which maintainers are explicitly requesting external contributions. |
| `duplicate` | An issue that is equivalent to an existing issue. |
| `invalid` | An issue that is out of scope, not actionable, or not applicable. |
| `wontfix` | A documented decision not to implement or address the issue. |

## Operational classification

The labels serve different purposes:

- **Initial issue types:** `bug`, `documentation`, `enhancement`, and `question`.
- **Contribution indicators:** `good first issue` and `help wanted`.
- **Triage outcomes:** `duplicate`, `invalid`, and `wontfix`.

Triage outcome labels are applied by maintainers after reviewing an issue. They are not exposed as issue templates because they do not represent categories that contributors should select when opening a new issue.

The `good first issue` and `help wanted` forms are intended primarily for maintainers or collaborators defining work for external contributors.

## Minimum workflow

1. The author selects the appropriate form from the **New issue** page.
2. The form automatically applies the corresponding initial label.
3. A maintainer reviews the issue for completeness, relevance, reproducibility, and possible duplication.
4. The maintainer may adjust labels, request additional information, or add `good first issue` or `help wanted` when appropriate.
5. When an issue is closed, the closing comment should briefly explain the decision or reference the pull request, commit, or issue that resolved it.

## Triage guidance

- Apply `duplicate` and link the canonical issue when the same problem or request is already tracked.
- Apply `invalid` when the issue is out of scope, not actionable, based on unsupported assumptions, or lacks required information after a reasonable request for clarification.
- Apply `wontfix` only when maintainers have made a deliberate decision not to proceed; record the rationale in the issue before closing it.
- Remove obsolete or conflicting labels as the issue classification changes.

## Conventions

- Use concise, descriptive titles.
- Do not include credentials, access tokens, personal data, or other sensitive information.
- Link related issues, pull requests, commits, and documentation.
- Update the issue when its scope, status, or implementation decision changes.
- Keep technical discussion and final decisions in the issue whenever possible so that the project history remains traceable.
