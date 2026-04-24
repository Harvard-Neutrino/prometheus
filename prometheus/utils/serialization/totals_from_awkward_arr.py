import awkward as ak
import numpy as np

class IncompaticleFieldsError(Exception):
    """Error raised when two ``awkward.Array`` objects cannot be combined because fileds don't match."""
    def __init__(self, fields1, fields2):
        self.message = f"If `fields` not provided, array fields must fully overlap."
        super().__int__(self.message)

def join_awkward_arrays(arr1, arr2, fields=None):
    """Concatenate two ``awkward.Array`` objects event-by-event along shared fields.

    Parameters
    ----------
    arr1 : awkward.Array
        First array to join.
    arr2 : awkward.Array
        Second array to join.
    fields : list of str, optional
        Fields to join. If not provided, fields are inferred from the arrays
        and must fully overlap between ``arr1`` and ``arr2``.

    Returns
    -------
    arr : awkward.Array
        Array with the same fields, where each event contains the concatenation
        of corresponding events from ``arr1`` and ``arr2``.

    Raises
    ------
    IncompaticleFieldsError
        Raised if ``fields`` is not provided and the fields of ``arr1`` and
        ``arr2`` do not fully overlap.
    """
    # Infer fields from arrs if not passed
    if fields is None:
        if not (
            set(arr1.fields).issubset(set(arr2.fields)) and
            set(arr2.fields).issubset(set(arr1.fields))
        ):
            raise IncompaticleFieldsError(arr1.fields, arr2.fields)
        else:
            fields = arr1.fields

    arr = ak.Array(
        {
            k: [np.hstack([x, y]) 
            for x, y in zip(getattr(arr1, k), getattr(arr2, k))]
            for k in fields
        }
    )

    return arr

def totals_from_awkward_arr(
    arr
):
    """Combine per-particle hit arrays from an ``awkward.Array`` into a single total array.

    Parameters
    ----------
    arr : awkward.Array
        Array with per-particle fields (any field that is not ``event_id``,
        ``mc_truth``, or ``total`` is treated as a particle field).

    Returns
    -------
    outarr : awkward.Array or None
        Combined array across all particle fields, or ``None`` if no particle
        fields are present.
    """
    # These are the keys which refer to the physical particles
    particle_fields = [
        field for field in arr.fields
        if field not in "event_id mc_truth total".split()
    ]

    # Return `None` if no particles made light
    if len(particle_fields)==0:
        return None

    outarr = getattr(arr, particle_fields[0])
    for field in particle_fields[1:]:
        outarr = join_awkward_arrays(outarr, getattr(arr, field))
    return outarr
