---
name: docstrings-style
description: Enforces style rules for docstrings. Use when checking or generating docstrings.
---

# Format docstrings in NumPy style 

## When to Use This Skill

Use this skill whenever the user asks to "generate docstrings", "normalize docstrings", "convert docstrings to NumPy style", "make docstrings consistent with style standards", "check docstrings", or otherwise clean up Python docstrings.

Use it when generating docstrings along with generating code as well.

Do **not** apply these rules to pure code identifiers unless explicitly allowed below.

## Instructions
  
1. Use the project-specific styling rules defined in the [text-style-prometheus](../text-style-prometheus/SKILL.md) skill.

2. Keep the new/edited docstrings **consistent with existing docstrings** in formatting, styling and type spelling.

3. **Preserve content and tone in existings docstrings**
  - When editing docstrings, do **not** change the high-level meaning or intent of the text.
  - Preserve informal or quirky phrasing where possible.

4. Use the NumPy docstring style guide: <https://numpydoc.readthedocs.io/en/latest/format.html>
  - Keep or add a short one-line summary, sentence-cased and ending with a period.
  - Use these section headers when relevant:
    - `Parameters` for functions, `Attributes` for classes.
    - `Returns`
    - `Raises`
    - `Yields`
    - `Notes`
    - `Examples`
    - `See Also`
  - Format section headers **exactly** like:
    - `Section Header Name`
      `-------------------`

5. **Summary line length**
When generating docstrings, **prefer shorter sentences** for the summary lines to comply with the 75-characters line length limit, recommended by NumPy docstring style guide.

If you have to use a longer sentence that will exceed the character limit, or if you are editing an existing docstring with a long summary line, **do not break the line with a new line**. If you do, the documentation generation tool we use will not render the line correctly on the documentation website.

Examples:

- ❌ Avoid:
```python
"""Simulate the propagation of a particle and of any photons resulting from 
    the energy losses of this particle."""
# The part after the line break ("the energy losses of this particle") does not render on the documentation site
```

- ✅ Prefer:
```python
"""Simulate the propagation of a particle and of any photons resulting from the energy losses of this particle."""
# The entire line renders on the website.
```

6. **Parameter entries**
  - For each parameter, use the pattern:
    - `name : type` or `name : type, optional`
  - Put the description on the next indented line(s).
  - When editing existing docstrings:
    - If the parameter type is not defined, but is obvious from annotations or usage, include it (e.g. `list of Module`, `np.ndarray`).
    - For optional arguments already documented as optional (e.g. `rng` where default is set), add `, optional` to the type when appropriate.
 
7. **Returns entries**
  - If there is a single return value, use:
    - `name : type`
      `<indented sentence describing the value, starting with a capital letter and ending with a period.>`
  - If the function returns a bare value (e.g. `Detector`), you may use a short name like `det` or reuse the informal name already in the project (`detector`, `modules`, etc.).
 
8. **Raises entries**
  - For each exception, use this format:
    ```
    ExceptionType
      <indented sentence starting with "Raised if ..." or similar, capitalized and ending with a period.>
    ```

    Example:

    ```python
    """
    Raises
    ------
    IncompatibleSerialNumbersError
        Raised if serial numbers length doesn't match number of DOMs.
    """
    """
    ```
  - Prefer "Raised if ..." or "Raised when ..." phrasing.

9. **Descriptions within section blocks start with a capital letter and end with a period**

Within each Parameters / Returns / Raises / Notes / other section block, ensure every description starts with a capital letter and ends with a period.

Examples:
- Good: `x : float` / `    X-position of the line.`
- Avoid leading lowercase descriptions (e.g. `length of ...`); convert to `Length of ...`.

10. **Enclose code references in double backticks** 

When referencing anything code-related inside docstrings: variables, expressions, class names etc., enclose them in double backticks(``variable``).

For example:

```python
def t_geo(x, t_0, direc, x_0):
  """
  Calculate the expected arrival time of unscattered photons at position ``x``, emitted by a muon with direction ``direc`` and time ``t_0`` at position ``x_0``.

  Parameters
  ----------
  x : (3,1) np.ndarray
      Position of the sensor.
  t_0 : float
      Time at which the muon is at ``x_0``.
  direc : (3,1) np.ndarray
      Normalized direction vector of the muon.
  x_0 : (3, 1) np.ndarray
      Position of the muon at time ``t_0``.
  """
```

11. **Do not rename parameters or change function signatures**

When editing docstrings, renaming parameters and otherwise deviating from what is already written, is only acceptable if there is a discrepancy between the code and the docstring that describes the code. 

For example:
  - A parameter is present in the function signature, but not in the docstring.
  - Parameter types in a docstring differ from the function signature.

In these cases, **treat the code as the source of truth** and edit the docstring to match it.

For example, in this case:

```python
def add(a:int, b:int, c:int)
  """A function to add two numbers.
  
  Parameters
  ----------
  first_one : int
    First thing to add.
  second_one : str
    Second thing to add.
  """
```

The docstring should be edited like so:

```python
def add(a:int, b:int, c:int)
  """A function to add three numbers.
  
  Parameters
  ----------
  a : int
    First number to add.
  b : int
    Second number to add.
  c : int
    Third number to add.
  """
```

## Good docstring examples from this project
 
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
    """Write detector coordinates into f2k format.
 
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