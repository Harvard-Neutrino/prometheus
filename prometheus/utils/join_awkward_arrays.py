import awkward as ak
import numpy as np

from .serialization.totals_from_awkward_arr import IncompatibleFieldsError


def join_awkward_arrays(arr1, arr2, fields=None):
    """Concatenate two ``awkward.Array`` objects event-by-event along shared fields.

    Parameters
    ----------
    arr1 : ak.Array
        First array to join.
    arr2 : ak.Array
        Second array to join.
    fields : list of str, optional
        Fields to join. If not provided, the fields are inferred from the arrays
        and must fully overlap between ``arr1`` and ``arr2``.

    Returns
    -------
    arr : ak.Array
        Array with the same fields, where each event contains the concatenation
        of the corresponding events from ``arr1`` and ``arr2``.
    """
    # Infer fields from arrs if not passed
    if fields is None:
        if not (
            set(arr1.fields).issubset(set(arr2.fields))
            and set(arr2.fields).issubset(set(arr1.fields))
        ):
            raise IncompatibleFieldsError(arr1.fields, arr2.fields)
        else:
            fields = arr1.fields

    arr = ak.Array(
        {k: [np.hstack([x, y]) for x, y in zip(getattr(arr1, k), getattr(arr2, k))] for k in fields}
    )

    return arr
