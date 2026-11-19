"""
Image dimension manipulation utilities.
"""



def get_indices_for_strings(strings):
    """
    Find which dimension index corresponds to dimension name strings.

    The default used order of dimensions is [t, z, x, y, r, c].

    Parameters
    ----------
    strings : str or list of str
        String dimension names or list of string dimension names to convert to indices.

    Returns
    -------
    list of int or None
        List of indices corresponding to the input strings.
    """
    fixed_order = ["t", "z", "x", "y", "r", "c"]
    # Define a mapping for shorthand versions to the full versions
    shorthand_mapping = {
        "time": "t",
        "fid": "t",
        "spectral": "t",
        "spectroscopic": "t",
        "echoes": "t",
        "sampling": "t",
        "spec": "t",
        "repetitions": "r",
        "reps": "r",
        "x-axis": "x",
        "z-axis": "z",
        "y-axis": "y",
        "chan": "c",
        "ch": "c",
        "chans": "c",
        "channels": "c",
        "channel": "c",
        "acquisition_idx": "t",
        "repetition_idx": "r",
        "slice_idx": "z",
        "phase_idx": "y",
        "channel_idx": "c",
        "read_idx": "x",
    }

    if isinstance(strings, str):
        strings = [strings]

    indices = []
    for s in strings:
        if isinstance(s, str):
            full_version = shorthand_mapping.get(s, s)
            try:
                index = fixed_order.index(full_version)
                indices.append(index)
            except ValueError:
                indices.append(None)
        elif isinstance(s, int):
            indices.append(s)
        else:
            indices.append(None)
    return indices
