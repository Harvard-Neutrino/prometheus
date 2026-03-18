---
name: text-style-prometheus
description: >
  Enforces Prometheus-specific text style rules for docstrings, markdown, and other prose. Use when checking or generating any non-code text (docstrings, .md files, comments-as-prose).
---

# Prometheus Text Style

## When to Use This Skill

Use this skill whenever you:

- Review or generate **Python docstrings**.
- Edit or create **markdown** (`.md`) or other documentation.
- Write or normalize **comments that read as prose**.
- Produce any **user-facing text** (error messages, logs, CLI help, README sections).

Do **not** apply these rules to pure code identifiers unless explicitly allowed below.

## Wording and Grammar Preferences

### 1. `Create` or `build` VS `make` in code-related statements and comments

When describing code (functions, methods, classes etc.), prefer the words `create` or `build` to the word `make`, since `make` is often used as a designated term related to build tools in programming.

- ❌ `Make a list of PROPOSAL density distributions`
- ✅ `Create a list of PROPOSAL density distributions`
- ❌ `Make a PROPOSAL propagator`
- ✅ `Build a PROPOSAL propagator`

### 2. Use present tense in code documentation and docstrings

When describing what classes, methods and other pieces of code do, prefer present tense. For example:

- ❌ `lepton_propagator: Prometheus LeptonPropagator object which will be used to generate losses from the particles.`
- ✅ `lepton_propagator: Prometheus LeptonPropagator object which **is** used to generate losses from the particles.`

## Core Terminology Rules

### 1. `Prometheus` project name

- Always spell as **`Prometheus`** (capitalized).
- Do not change the form in prose:
  - ✅ `Prometheus`
  - ❌ `prometheus`, `PROMETHEUS`, `P.P.C.`, or other variants.

### 2. `ppc` library name

- Always spell the library name as **`ppc`** (all lowercase).
- Do not change the form in prose:
  - ✅ `ppc`
  - ❌ `PPC`, `Ppc`, `P.P.C.`, or other variants.

If a code identifier already uses a different capitalization (e.g. a class named
`PPCConfig`), **leave the code identifier as-is**, but still refer to the library as
`ppc` in surrounding prose.

### 3.`GENIE` neutrino event generator name

- Always spell as **`GENIE`** (all uppercase).
- Do not change the form in prose:
  - ✅ `GENIE`
  - ❌ `genie`, `Genie`, or other variants.

### 4.`PROPOSAL` tool / programme name

- Always spell as **`PROPOSAL`** (all uppercase).
- When referred to a tool, do not change the form in prose:
  - ✅ `PROPOSAL`
  - ❌ `proposal`, `Proposal`, or other variants.
- If the word "proposal" used as a regular word meaning a plan or suggestion put forward for consideration or discussion by others, then there is no need to capitalize, style as you would any other word.

### 5. `geo file` (Geometry File)

- In prose, always write **`geo file`** (two words, lowercase).
- Avoid: ❌ `geofile`, `GeoFile`, or other variants.

#### Code vs Prose Exception

- The single-word form **`geofile`** is acceptable **only in code contexts**, such as:
  - Variable, function, class, or module names (e.g. `load_geofile`, `GeofileError`).
  - Configuration keys, or API fields that are explicitly part of a code
    interface (e.g. JSON key `"geofile"`).
- When explaining these identifiers in text, prefer:
  - "The `geofile` parameter specifies the **geo file** to load."

## Application Guidelines

Apply these rules to:
  - Docstrings and comments written as sentences.
  - Markdown and other documentation.
  - User-facing messages, logs, or CLI help text.

When editing or generating text:

1. **Scan for terminology**:
   - Normalize any mention of the library to `ppc`.
   - Normalize any prose mention of the geometry file to `geo file`.
2. **Respect code identifiers**:
   - Do not rename variables, functions, classes, or modules solely to satisfy these
     prose rules.
   - Only adjust the *surrounding text* unless the user explicitly asks to rename code.
3. **Be consistent**:
   - Use the chosen forms (`ppc`, `geo file`) uniformly within a document or docstring.

If a code identifier already uses a different capitalization (e.g. a class named
`PPCConfig`), **leave the code identifier as-is**, but still refer to the term/name in surrounding prose as defined by the above rules.

## Examples

### Example 1: Docstring

Before:

```python
def load_geofile(path: str) -> GeoData:
    """
    Load the GeoFile used by PPC.
    """
```

After (applying this skill):

```python
def load_geofile(path: str) -> GeoData:
    """
    Load the geo file used by ppc.
    """
```

### Example 2: Markdown

Before:

```markdown
The PPC library reads GeoFiles from disk.
```

After:

```markdown
The ppc library reads geo files from disk.
```

