# AGENTS.md - Instructions for Coding Assistant LLMs

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Kotlin](https://img.shields.io/badge/Kotlin-7F52FF?logo=kotlin&logoColor=white)](https://kotlinlang.org/)
[![Java](https://img.shields.io/badge/Java-21-ED8B00?logo=openjdk&logoColor=white)](https://www.java.com/)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Go](https://img.shields.io/badge/Go-1.22+-00ADD8?logo=go&logoColor=white)](https://go.dev/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)

> **Version**: 1.0
> **Last Updated**: 2026-07-30
> **Purpose**: Authoritative reference for AI assistants (Claude, GPT, Gemini, Copilot, etc.) working in repositories generated from this template.

## Table of Contents

1. [Project Overview & Mission](#1-project-overview--mission)
2. [Technical Stack & Governance](#2-technical-stack--governance)
3. [Module Boundaries](#3-module-boundaries)
4. [Key CLI Entry Points](#4-key-cli-entry-points)
5. [Coding Standards](#5-coding-standards)
6. [Known Constraints](#6-known-constraints)

## 1. Project Overview & Mission

> **TODO:** Replace with a one-paragraph description of what the generated project does and why it exists.

This repository is a scaffold, not a product. When it is used via "Use this template", update this section first — every other document under `.agent/` and `docs/` links back to it.

## 2. Technical Stack & Governance

| Component | Specification | Notes |
| --- | --- | --- |
| Python | 3.11+ | Managed via `uv`; always `source .venv/bin/activate` |
| TypeScript | 5 | Managed via `npm` workspaces |
| Kotlin | 2.0 / JVM 21 | Built via Gradle (`gradlew`) |
| Java | 21 | Built via Maven (`mvn`) |
| Rust | stable | Managed via `cargo` |
| Go | 1.22+ | Managed via `go.mod`, no external tooling required |
| C++ | 17 | Built via CMake, environment managed by Pixi or vcpkg |
| Config | `.env` / `configs/` | Environment-specific values never committed; see `.env.example` |

## 3. Module Boundaries

- `python/src` — domain logic. No imports from `typescript/`, `kotlin/`, `java/`, or other language modules.
- `typescript/src` — presentation/CLI layer. Talks to other modules only through their published APIs (HTTP, FFI, or CLI), never by reaching into their source trees.
- `rust/src` and `cpp/src` — performance-critical cores, exposed to higher-level languages through explicit bindings (e.g. `pyo3`/`pybind11`, `napi`, JNI). No language-specific logic should leak across the binding boundary.
- `go/cmd` and `go/internal` — services/CLIs. `internal/` is never imported from outside the `go/` module.
- `kotlin/src` — JVM/Android-facing code, isolated behind Gradle module boundaries.
- `java/src` — JVM-facing code, isolated behind Maven module boundaries.
- Cross-module contracts (schemas, protobufs, OpenAPI specs) live under `docs/` or a dedicated `schemas/` directory — never duplicated per language.

## 4. Key CLI Entry Points

| Command | Purpose |
| --- | --- |
| `just --list` | List all available command-runner recipes |
| `just test` | Run the test suite for every language module |
| `just lint` | Run linters/formatters for every language module |
| `just docs` | Build the MkDocs + Sphinx documentation site |
| `just docker-up` | Start the local Docker Compose stack |

## 5. Coding Standards

- Follow the per-language rules in [`.agent/rules/`](rules/) (`python.md`, `typescript_react.md`, `kotlin.md`, `java.md`, `rust.md`, `go.md`, `cpp.md`).
- Prefer small, reviewable diffs. Do not reformat files unrelated to the change.
- Every new public function/class needs a docstring/doc-comment; every new module needs at least one test.
- Never commit secrets. Use `.env` (git-ignored) and document new variables in `.env.example`.

## 6. Known Constraints

> **TODO:** Document real constraints (rate limits, hardware requirements, licensing restrictions, etc.) once the project has them.

- This template repository does not build or run as-is — each language module contains only illustrative examples.
