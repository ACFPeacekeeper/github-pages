# Contributing to github-pages

[![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/ACFHarbinger/github-pages/actions/workflows/ci.yml/badge.svg)](https://github.com/ACFHarbinger/github-pages/actions/workflows/ci.yml)

Thank you for your interest in contributing! This document covers setup, style, and the PR process for this repository.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Development Setup](#2-development-setup)
3. [Code Style Guidelines](#3-code-style-guidelines)
4. [Git Workflow](#4-git-workflow)
5. [Pull Request Process](#5-pull-request-process)
6. [Testing Requirements](#6-testing-requirements)
7. [Issue Reporting](#7-issue-reporting)

---

## 1. Getting Started

### 1.1 Prerequisites

- Git, [Node.js](https://nodejs.org/) >= 20, `npm`.
- For `notebooks/`: `python` (>= 3.11) + [`uv`](https://github.com/astral-sh/uv).
- `pre-commit` (`pip install pre-commit && pre-commit install`).

### 1.2 Clone and bootstrap

```bash
git clone https://github.com/ACFHarbinger/github-pages.git
cd github-pages
npm install
npm run dev
```

## 2. Development Setup

The site lives at the repo root (`app/`, `src/`, `lib/`) as a single Next.js app; `notebooks/` is a separate `uv`-managed Python workspace (`cd notebooks && uv sync --extra dev`).

## 3. Code Style Guidelines

Follow the rules in [`.agent/rules/`](../.agent/rules/), primarily [`typescript_react.md`](../.agent/rules/typescript_react.md) and [`python.md`](../.agent/rules/python.md). Linting/formatting runs automatically via `.pre-commit-config.yaml` — run `pre-commit run --all-files` before pushing.

## 4. Git Workflow

- Branch from `main`: `feature/<short-description>` or `fix/<short-description>`.
- Keep commits focused; write commit messages that explain *why*, not just *what*.
- Rebase onto `main` before opening a PR.

## 5. Pull Request Process

1. Fill out the [PR template](../.github/PULL_REQUEST_TEMPLATE.md) in full.
2. Ensure CI is green (`npm run lint && npm test && npm run build`).
3. Request review; address feedback with new commits (don't force-push during review).
4. Squash-merge once approved.

## 6. Testing Requirements

New components/logic get a Jest test; new user-facing flows get a Cypress spec. See [`.agent/rules/test_writing.md`](../.agent/rules/test_writing.md).

## 7. Issue Reporting

Use the issue templates under [`.github/ISSUE_TEMPLATE/`](../.github/ISSUE_TEMPLATE/) — they help both humans and coding agents triage faster.
