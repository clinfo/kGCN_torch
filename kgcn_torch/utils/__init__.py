from pathlib import Path

def to_path(path: str) -> Path:
    """ change string to absolute path.
    """
    return Path(path).expanduser().resolve()
