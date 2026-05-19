from __future__ import annotations

from pathlib import Path


def validate_output_root_path(output_root) -> Path:
    output_root_text = str(output_root)
    stripped_output_root = output_root_text.strip()

    if output_root_text != stripped_output_root:
        raise ValueError(
            "output_root contains leading or trailing whitespace. "
            f"Received {output_root_text!r}. "
            "Check the command or Slurm script and remove the extra spaces."
        )

    return Path(output_root_text)
