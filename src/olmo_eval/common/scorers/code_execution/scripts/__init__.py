"""Python programs that are staged into a sandbox and run there.

These are loaded as text rather than imported: they execute inside the
container, against its interpreter, not in the harness process.
"""

from importlib import resources


def get_script(name: str) -> str:
    """Load a sandbox script by name.

    Args:
        name: Script name without the .py extension (e.g., "livecodebench_grader").

    Returns:
        The script source as a string.
    """
    return resources.files(__package__).joinpath(f"{name}.py").read_text()
