"""Input acquisition utilities for the Travel Order Resolver.

This module defines the interface used by the pipeline to obtain the
raw text describing a travel order. The goal is to decouple the source
of the text (keyboard input, file, or speech-to-text) from the rest of
the processing pipeline.
"""


def get_input_text() -> str:
    """Obtain the raw input text for a travel order.

    Returns
    -------
    str
        The sentence provided by the user that should be interpreted as
        a potential travel request.

    Notes
    -----
    The current project stage does not commit to a concrete input
    source. Future iterations may plug in different acquisition
    mechanisms (e.g. terminal input or speech-to-text) while keeping
    this interface stable.
    """
    raise NotImplementedError("Input acquisition is not implemented yet.")

