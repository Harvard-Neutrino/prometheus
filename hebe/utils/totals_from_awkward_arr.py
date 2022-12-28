import awkward as ak
import numpy as np

class IncompaticleFieldsError(Exception):
    """Raised when two awkward.Array cannot be combined because fields don't match"""
    def __init__(self, fields1, fields2):
        self.message = f"If `fields` not provided, array fields must fully overlap."
        super().__int__(self.message)

def join_awkward_arrays(arr1, arr2, fields=None):
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
