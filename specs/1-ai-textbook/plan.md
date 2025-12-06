# Implementation Plan: textbook-generation

**Branch**: `1-ai-textbook` | **Date**: 2025-12-04 | **Spec**: specs/1-ai-textbook/spec.md
**Input**: Feature specification from `/specs/1-ai-textbook/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This plan outlines the implementation for building an AI-native textbook with an integrated RAG chatbot. The core value lies in providing educational content enhanced with AI assistance, allowing learners to query the textbook content directly. The solution will leverage Docusaurus for the textbook UI and a FastAPI backend with Qdrant and Neon for the RAG chatbot, adhering to free-tier and minimal embedding constraints.

## Technical Context

**Language/Version**: TypeScript (for Docusaurus/React frontend), Python 3.11+ (for FastAPI backend)
**Primary Dependencies**: Docusaurus v3+, React, Qdrant, Neon (PostgreSQL), FastAPI, free-tier embedding models (e.g., MiniLM, Sentence Transformers)
**Storage**: Neon (PostgreSQL for vector database/metadata)
**Testing**: Frontend: Jest/React Testing Library (unit/integration), Cypress/Playwright (E2E). Backend: Pytest (unit/integration/API).
**Target Platform**: Web (GitHub Pages deployment)
**Project Type**: Hybrid (Docusaurus static site + FastAPI microservice)
**Performance Goals**: Fast build times (Docusaurus), responsive RAG chatbot answers (sub-5s latency for typical queries)
**Constraints**: No heavy GPU usage, minimal embeddings footprint, free-tier architecture for all components, RAG answers strictly from book text.
**Scale/Scope**: 6 short chapters, support for lightweight embeddings, optional Urdu translation, optional personalize chapter.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Specification-First**: All chapters and components are defined in specs before implementation. (`specs/1-ai-textbook/spec.md` exists)
- [x] **Living Documentation**: The book is treated as software; verified by builds, not just proofreading. (`npm run build` as success criteria)
- [x] **Educational Clarity**: Content must be accessible to technical learners (clear "How-to" focus). (Addressed during content creation)
- [x] **Automation**: Deployment to GitHub Pages must be automated via GitHub Actions. (Will be implemented)
- [x] **Format**: Docusaurus MDX (Markdown + React) with correct Frontmatter. (Technical requirement)
- [x] **Naming Convention**: Use numbered prefixes (e.g., 01-intro, 02-setup) for automatic sidebar ordering. (Technical requirement)
- [x] **Code Integrity**: All code snippets in the book must be syntactically correct and runnable. (Technical requirement)
- [x] **Content Depth**: Minimum 3 distinct sections per chapter (Introduction, Core Concept, Practical Example). (Content creation guideline)
- [x] **Admonitions**: Use Docusaurus admonitions (Note, Tip, Warning) for key takeaways. (Content creation guideline)
- [x] **Tech Stack**: Docusaurus v3+, React, TypeScript, GitHub Pages. (Technical requirement)
- [x] **Build Requirement**: npm run build must pass with zero broken links. (Success criteria)
- [x] **Deployment**: Must use gh-pages branch or GitHub Actions workflow. (Technical requirement)
- [x] **Styling**: Use standard Docusaurus theme + Swizzling only if specified in specs. (Technical requirement)
- [x] **Tools**: Spec-Kit Plus workflow with Claude Code or Gemini CLI. (Process guideline)

## Project Structure

### Documentation (this feature)

```text
specs/1-ai-textbook/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
website/ (Docusaurus frontend)
├── src/
│   ├── pages/        # Main content pages, e.g., index.js, about.js
│   ├── components/   # Reusable React components
│   ├── theme/        # Swizzled Docusaurus theme components (if needed)
│   └── data/         # Static data, configuration
├── docs/             # Markdown files for chapters, auto-generated sidebar
├── blog/             # Optional blog posts
├── static/           # Static assets (images, CSS)
└── docusaurus.config.js # Docusaurus configuration

rag-backend/ (FastAPI backend)
├── app/
│   ├── main.py       # FastAPI application entry point
│   ├── api/          # API endpoints
│   ├── services/     # Business logic, RAG implementation
│   ├── models/       # Pydantic models for request/response
│   └── utils/        # Utility functions (e.g., embedding generation)
├── tests/
│   ├── unit/
│   └── integration/
└── requirements.txt  # Python dependencies

.github/workflows/    # GitHub Actions for CI/CD
├── build-deploy.yml  # Docusaurus build and deployment to GitHub Pages
└── rag-ci.yml        # RAG backend CI/CD (optional, if separate deploy)

```

**Structure Decision**: The project will adopt a monorepo-like structure with `website/` for the Docusaurus frontend and `rag-backend/` for the FastAPI RAG service. This separation allows for independent development, deployment, and scaling of the textbook UI and the RAG functionality.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |
