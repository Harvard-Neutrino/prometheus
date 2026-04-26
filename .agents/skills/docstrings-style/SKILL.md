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
  
1. Use the NumPy docstring formatting rules defined in the [numpy-docstring-format](../numpy-docstring-format/SKILL.md) skill.

2. Use the project-specific text styling rules defined in the [text-style-prometheus](../text-style-prometheus/SKILL.md) skill.

3. Keep the new/edited docstrings **consistent with existing docstrings** in formatting, styling and type notation.

4. **Preserve content and tone in existing docstrings**
  - When editing docstrings, do **not** change the high-level meaning or intent of the text.
  - Preserve informal or quirky phrasing where possible. 

5. **Keep types in docstrings consistent with imports and typehints**

- Variable types in docstrings should exactly match the typehints, if available.
  Example:
  ```python
  def some_method(func: Callable)
  """This method does a thing.

  Parameters
  ----------
  func : Callable
    A function that helps this method do a thing.
  ```
- If typehints are not available, refer to the Python documentation for type names: https://docs.python.org/3/library/stdtypes.html 

- Types from imported external libraries should also be consistent with typehints, the names of external libraries should be consistent with how they are imported. Examples:
  - Good:
    ```python
    import numpy as np
    import proposal as pp

    def some_method(x: np.ndarray, y: pp.particle)
      """A method that does a thing.

      Parameters
      ----------
      x : np.ndarray
        Description of ``x``.
      y : pp.particle
        Description of ``y``.
    ```
  - Avoid:
    ```python
    import numpy as np
    import proposal as pp

    def some_method(x: np.ndarray, y: pp.particle)
      """A method that does a thing.

      Parameters
      ----------
      x : numpy.ndarray
        Description of ``x``.
      y : proposal.particle
        Description of ``y``.
    ```

6. **Do not rename parameters or change function signatures**

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
    First thing to add.
  b : int
    Second thing to add.
  c : int
    Third thing to add. 
  """
```

In the edited dosctring, the parameters have types matching typehints in function signature, and the description of the missing parameter was added.

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
        hexadecimal format, but their exact value does not matter. If
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