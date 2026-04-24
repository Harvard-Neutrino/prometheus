import numpy as np

class SimpleDataset(object):
    """Simple Dataset that returns a tuple of arrays (inputs, outputs).

    Parameters
    ----------
    *arrays : sequence of array-like
        Arrays provided to the dataset; the first array determines dataset length.

    Attributes
    ----------
    _arrays : tuple
        Stored input/output arrays.
    _len : int
        Number of samples.
    """

    def __init__(self, *arrays):
        """Initialize the dataset with one or more aligned arrays.

        Parameters
        ----------
        *arrays : sequence of array-like
            Arrays to store; all arrays must have the same length.
        """
        super(SimpleDataset, self).__init__()
        self._arrays = arrays

        self._len = len(arrays[0])

        for arr in arrays:
            if len(arr) != self._len:
                raise ValueError("Inputs and outputs must have same length.")

    def __getitem__(self, idx):
        """Return one or more arrays for the given index or slice.

        Parameters
        ----------
        idx : int or slice or array-like
            Indexing expression selecting samples.

        Returns
        -------
        list
            List of arrays (at least 2-D) corresponding to the requested index.
        """
        if isinstance(idx, int):
            idx = [idx]

        outs = [np.atleast_2d(arr[idx]) for arr in self._arrays]

        return outs

    def __len__(self):
        """Return number of samples in the dataset."""
        return self._len


class SubSet(object):
    """
    Dataset subset.

    Parameters
    ----------
    dataset : object
        Original dataset.
    subset_ix : sequence of int
        Indices included in the subset.

    Attributes
    ----------
    _subset_ix : sequence
        Stored subset indices.
    _len : int
        Number of elements in the subset.
    _dataset : object
        Reference to original dataset.
    """

    def __init__(self, dataset, subset_ix):
        """Create a subset view of an existing dataset.

        Parameters
        ----------
        dataset : object
            Original dataset object supporting ``__len__`` and ``__getitem__``.
        subset_ix : sequence of int
            Indices included in the subset.
        """
        super(SubSet, self).__init__()
        if max(subset_ix) > len(dataset):
            raise RuntimeError("Invalid index")

        self._subset_ix = subset_ix
        self._len = len(subset_ix)
        self._dataset = dataset

    def __len__(self):
        """Return number of elements in the subset."""
        return self._len

    def __getitem__(self, idx):
        """Return the item(s) from the original dataset at the subset indices.

        Parameters
        ----------
        idx : int or slice or array-like
            Indexing into the subset.
        """
        true_ix = self._subset_ix[idx]
        return self._dataset[true_ix]


def create_random_split(dataset, split_len, rng):
    """
    Create a random split.

    Parameters
    ----------
    dataset : object
        Dataset to split.
    split_len : int
        Length of the first split.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    tuple
        Tuple of (first_split, second_split) subsets.
    """
    ixs = np.arange(len(dataset))
    rng.shuffle(ixs)
    first_split = SubSet(dataset, ixs[:split_len])
    second_split = SubSet(dataset, ixs[split_len:])

    return first_split, second_split


def randomize_ds(dataset, rng):
    """
    Randomize a dataset.

    Parameters
    ----------
    dataset : object
        Dataset to randomize.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    SubSet
        Randomized subset of the dataset.
    """
    ixs = np.arange(len(dataset))
    rng.shuffle(ixs)
    return SubSet(dataset, ixs)


def downsample_ds(dataset, fraction, rng, copy=False):
    """Return a downsampled subset of the dataset.

    Parameters
    ----------
    dataset : object
        Dataset to sample.
    fraction : float
        Fraction of dataset to include (0 < fraction <= 1).
    rng : numpy.random.Generator
        Random number generator.
    copy : bool, optional
        If True, return an array slice of the dataset instead of a SubSet.

    Returns
    -------
    SubSet or array-like
        Downsampled dataset view or copy.
    """

    subset_len = int(fraction * len(dataset))
    ixs = np.arange(len(dataset))
    rng.shuffle(ixs)
    if copy:
        return dataset[ixs[:subset_len]]

    return SubSet(dataset, ixs[:subset_len])


class DataLoader(object):

    """
    Iterator over a dataset yielding batches.

    Parameters
    ----------
    dataset : object
        Dataset providing ``__len__`` and ``__getitem__``.
    batch_size : int
        Batch size.
    rng : numpy.random.Generator
        Random number generator.
    shuffle : bool, optional
        Whether to shuffle dataset each epoch.
    infinite : bool, optional
        If True, iterator yields batches indefinitely.

    Attributes
    ----------
    _n_batches : int
        Number of batches per epoch.
    """

    def __init__(self, dataset, batch_size, rng, shuffle=False, infinite=False):
        """Create a data loader that yields batches from ``dataset``.

        Parameters
        ----------
        dataset : object
            Dataset providing ``__len__`` and ``__getitem__``.
        batch_size : int
            Number of samples per batch.
        rng : numpy.random.Generator
            Random number generator used for shuffling.
        shuffle : bool, optional
            Whether to shuffle the dataset each epoch.
        infinite : bool, optional
            If True, the iterator yields batches indefinitely.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._rng = rng
        self._shuffle = shuffle
        self._n_batches = int(np.ceil(len(self._dataset) / self._batch_size))
        self._infinite = infinite

    def __iter__(self):
        """Yield batches from the dataset.

        Yields
        ------
        array-like
            Batches of data retrieved via ``dataset[ixs]``.
        """
        while True:
            if self._shuffle:
                ds = randomize_ds(self._dataset, self._rng)
            else:
                ds = self._dataset

            for batch in range(self._n_batches):
                upper = min(len(ds), (batch + 1) * self._batch_size)
                ixs = np.arange(batch * self._batch_size, upper)
                yield ds[ixs]

            if not self._infinite:
                break

    @property
    def n_batches(self):
        """Number of batches per epoch."""
        return self._n_batches


class StochasticLoader(object):
    """
    Loader that yields batches of unique random indices.

    Parameters
    ----------
    dataset : object
        Dataset to sample from.
    batch_size : int
        Size of each batch.
    rng : numpy.random.Generator
        Random number generator.
    """

    def __init__(self, dataset, batch_size, rng):
        """Initialize the stochastic loader.

        Parameters
        ----------
        dataset : object
            Dataset to sample from.
        batch_size : int
            Size of each returned batch.
        rng : numpy.random.Generator
            Random number generator.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._rng = rng
        # self._n_batches = int(np.ceil(len(self._dataset) / self._batch_size))

    def __iter__(self):
        """Yield batches of unique random indices from the dataset.

        This generator attempts to draw a batch with unique indices and will
        retry up to a small number of times before warning.
        """

        while True:
            is_unique = False
            cnt = 0
            while not is_unique:
                ixs = self._rng.randint(0, len(self._dataset), size=self._batch_size)
                is_unique = len(np.unique(ixs)) == self._batch_size
                cnt += 1
                if cnt == 20:
                    print("Trouble finding unique batch")
            yield self._dataset[ixs]
