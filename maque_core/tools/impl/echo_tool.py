"""Echo tool — a simple tool that echoes back whatever you send it.

Useful for testing and as a reference implementation.
"""

from ..decorator import tool


@tool
def echo(text: str) -> str:
    """Echo back the input text.

    Args:
        text: The text to echo back.

    Returns:
        The same text that was provided.
    """
    return text
