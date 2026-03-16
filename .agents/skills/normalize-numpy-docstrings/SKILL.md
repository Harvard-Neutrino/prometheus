---
name: normalize-numpy-docstrings
description: Normalize existing Python docstrings to NumPy style while preserving informal wording, especially in this project. Use when the user asks to normalize or clean up Python docstrings to NumPy style.
---

# Normalize NumPy docstrings

## When to use this skill

Use this skill whenever the user asks to:

- "normalize docstrings", "convert docstrings to NumPy style", or otherwise clean up Python docstrings.
- Apply the same normalization you previously used in `prometheus/detector` (this project).

Assume the code may use informal, quirky wording, and that wording should generally be preserved unless explicitly asked to change it.

## Instructions

When normalizing docstrings:

1. **Preserve content and tone**
   - Do **not** change the high-level meaning or intent of the text.
   - Preserve informal or quirky phrasing where possible.
   - You may fix obviously broken grammar and typos only when doing so does not change tone or meaning.
   - You may perform stylistic fixes based on rules defined in text-style-prometheus skill.

2. **Structure docstrings in NumPy style**
   - Keep or add a short one-line summary, sentence-cased and ending with a period.
   - Use these section headers when relevant:
     - `Parameters`
     - `Returns`
     - `Raises`
   - Format section headers **exactly** like:
     - `Parameters`
       `----------`
     - `Returns`
       `-------`
     - `Raises`
       `------`

3. **Parameter entries**
   - For each parameter, use the pattern:
     - `name : type` or `name : type, optional`
   - Put the description on the next indented line(s).
   - Start the description with a **capitalized word** and write as a proper sentence that ends with a period:
     - Good: `x : float` / `    X-position of the line.`
     - Avoid leading lowercase descriptions (e.g. `length of ...`) in final text; convert to `Length of ...`.
   - Keep existing explanations; only adjust capitalization and punctuation as needed.
   - If the type is obvious from annotations or usage, include it (e.g. `list of Module`, `numpy.ndarray`, `str or None`).
   - For optional arguments already documented as optional (e.g. `rng` where default is set), add `, optional` to the type when appropriate.

4. **Returns entries**
   - If there is a single return value, use:
     - `name : type`
       `<indented sentence describing the value, starting with a capital letter and ending with a period.>`
   - If the function returns a bare value (e.g. `Detector`), you may use a short name like `det` or reuse the informal name already in the project (`detector`, `modules`, etc.).
   - Normalize descriptions to start with a capital and end with a period, but keep the original phrasing otherwise:
     - Example from this project:
       - `det : Detector`
       - `    A rhombus detector.`

5. **Raises entries**
   - For each exception, use:
     - `ExceptionType`
       `<indented sentence starting with "Raised if ..." or similar, capitalized and ending with a period.>`
   - Pattern learned from this project:
     - `IncompatibleSerialNumbersError`
       `    Raised if serial numbers length doesn't match number of DOMs.`
     - `IncompatibleMACIDsError`
       `    Raised if MAC IDs length doesn't match number of DOMs.`
   - Prefer "Raised if ..." or "Raised when ..." phrasing.

6. **Consistency rules**
   - Within each `Parameters` / `Returns` / `Raises` block:
     - Ensure **every** description:
       - Starts with a capital letter.
       - Ends with a period.
   - Maintain consistent type spellings that you have already used in this project, e.g.:
     - `numpy.ndarray`
     - `tuple of int`
     - `list of Module`
     - `Medium or None`

7. **What NOT to change**
   - Do **not** rename parameters or change function signatures.
   - Do **not** rewrite long narrative text beyond light capitalization/punctuation/typos fixes.
   - Do **not** "formalize" jokes or intentionally informal comments unless the user asks you to.

## Examples from this project

- Constructor example:

```python
def __init__(self, modules: List[Module], medium: Union[Medium, None]):
    """Initialize detector.

    Parameters
    ----------
    modules : list of Module
        List of all the modules in the detector.
    medium : Medium or None
        Medium in which the detector is embedded.
    """
```

- Function with parameters, returns, and raises:

```python
def to_f2k(...):
    """Write detector corrdinates into f2k format.

    Parameters
    ----------
    geo_file : str
        File name of where to write it.
    serial_nos : list of str, optional
        Serial numbers for the optical modules. These MUST be in
        hexadecimal format, but there exact value does not matter. If
        nothing is provided, these values will be randomly generated.
    mac_ids : list of str, optional
        MAC (I don't think this is actually what this is called) IDs
        for the DOMs. By default these will be randomly generated. This
        is probably what you want to do.

    Raises
    ------
    IncompatibleSerialNumbersError
        Raised if serial numbers length doesn't match number of DOMs.
    IncompatibleMACIDsError
        Raised if MAC IDs length doesn't match number of DOMs.
    """
```

When asked to "normalize docstrings" in this repo in the future, follow the patterns and constraints above. If project-specific quirks appear (e.g. unconventional names or jokes), keep them, only wrapping them in consistent NumPy-style structure and capitalization. 

