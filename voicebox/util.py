"""Small helpers every module was re-typing: stepping through a list of
options, and cutting a label to fit a row."""


def cycle_index(options, current, d=1):
    """The entry after (d >= 0) or before `current` in `options`, wrapping.
    A `current` that is no longer in the list (an unplugged device, an
    uninstalled voice) restarts from the first entry. Empty list -> None."""
    if not options:
        return None
    try:
        i = options.index(current)
    except ValueError:
        i = 0
    return options[(i + (1 if d >= 0 else -1)) % len(options)]


def ellipsize(text, width):
    """`text` cut to `width` characters with a trailing "..." when needed."""
    text = str(text)
    return text if len(text) <= width else text[:max(1, width - 3)] + "..."
