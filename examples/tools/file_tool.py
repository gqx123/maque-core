"""File tool — read and write files on the local filesystem.

An example tool that demonstrates file I/O operations.
"""

from maque_core.tools.decorator import tool


@tool
def read_file(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: Path to the file to read.

    Returns:
        The file contents as a string.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file.

    Args:
        path: Path to the file to write.
        content: Content to write to the file.

    Returns:
        A confirmation message with the file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {len(content)} characters to {path}"
