"""加法工具：将两个数字相加并返回结果。"""

from ..decorator import tool


@tool(name="add", description="Add two numbers together and return the sum.")
def add_tool(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum of a and b.
    """
    return a + b
