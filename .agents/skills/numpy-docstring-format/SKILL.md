---
name: numpy-docstring-format
description: Formatting and styling rules for NumPy docstrings. Use when checking or generating docstrings.
---

# NumPy docstring style rules 

## When to Use This Skill

Use this skill whenever you review or generate **Python docstrings**.

## Instructions

1. Use the NumPy docstring style guide: <https://numpydoc.readthedocs.io/en/latest/format.html>.

2. Enclose docstrings in triple quotes (""") and always start the docstring text on the same line as opening quotes.
  - Single-line docstring example:
    ```python
    def some_method
    """The docstrings for this method."""
    ```
  - Multi-line docstring example:
    ```python
      def some_method
        """Short summary of the method.

        More details on what it does.
        """
    ```
  - Avoid:
  ```python
    def some_method
      """
      Summary starting on the next line.

      More details.
      """
  ```

3. Start docstrings with a short one-line summary, sentence-cased and ending with a period.

4. Use these section headers when relevant:

- `Parameters` for functions, `Attributes` for classes.
- `Returns`
- `Raises`
- `Yields`
- `Notes`
- `Examples`
- `See Also`

5. Format section headers with a capitalized section header name on one line, and the underlines matching the section header name length on the next line:
`Section Header Name`
`-------------------`

Examples:
```python
"""A docstring for a method.

Parameters
----------

Returns
-------

Raises
------

Notes
-----
"""
```

6. **Summary line length**

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

7. **Parameter entries**
  - For each parameter, use the pattern:
    - `name : type` or `name : type, optional`
  - Put the description on the next indented line(s).
  - When editing existing docstrings:
    - If the parameter type is not defined, but is obvious from annotations or usage, include it (e.g. `list of Module`, `np.ndarray`).
    - For optional arguments already documented as optional (e.g. `rng` where default is set), add `, optional` to the type when appropriate.

8. **Returns entries**
  - If there is a single return value, use:
    - `name : type`
      `<indented sentence describing the value, starting with a capital letter and ending with a period.>`
 
9. **Raises entries**
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
    ```
  - Prefer "Raised if ..." or "Raised when ..." phrasing.

10. **Notes entries**
  - Format the section exactly like:
      ```python
      """
      Notes
      -----
      Some additional information.

      Some more additional information.
      """
      ```
  - Notes block can use math expressions in LateX format:
    - Example: .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}
    - Expressions can be multi-line or inline

11. **Examples entries**
  - Examples entries are meant to illustrate the usage of the documented object.
  - Format examples in doctest format: https://docs.python.org/3/library/doctest.html
  - Format the section exactly like:
    ```python
    """
    Examples
    --------
    >>> np.add(1, 2)
    3

    Comment explaining the second example.
    >>> np.add([[1, 2], [3, 4]],
    ...        [[5, 6], [7, 8]])
    array([[ 6,  8],
           [10, 12]])
    """
    ```
  - If there is an existing docstring with an example of how to import a module, instantiate a class, etc. and it doesn't make sense to use the Examples section, use the Notes section instead.


12. **Descriptions within section blocks start with a capital letter and end with a period**

Within each Parameters / Returns / Raises / Notes / other section block, ensure every description starts with a capital letter and ends with a period.

Examples:
- Good: `x : float` / `    X-position of the line.`
- Avoid leading lowercase descriptions (e.g. `length of ...`); convert to `Length of ...`.

13. **Enclose code references in double backticks** 

When referencing anything code-related inside docstrings: variables, expressions, class names etc., enclose them in double backticks like so: ``variable``.

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

14. Enclose bare hyperlinks in angle brackets:

`Notes`
`-----`
`Read more: <https://harvard-neutrino.github.io/prometheus/>`
