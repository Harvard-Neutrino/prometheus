import awkward as ak

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
            k: [np.hstack([x, y]) for x, y in zip(getattr(arr1, k), getattr(arr2, k))]
        for k in fields}
    )

    return arr


