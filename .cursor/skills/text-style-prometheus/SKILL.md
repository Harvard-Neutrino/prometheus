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

## Core Terminology Rules

### 1. `ppc` Library Name

- Always spell the library name as **`ppc`** (all lowercase).
- Do not change the form in prose:
  - ✅ `ppc`
  - ❌ `PPC`, `Ppc`, `P.P.C.`, or other variants.

If a code identifier already uses a different capitalization (e.g. a class named
`PPCConfig`), **leave the code identifier as-is**, but still refer to the library as
`ppc` in surrounding prose.

### 2. `geo file` (Geometry File)

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

