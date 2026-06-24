from __future__ import annotations

from pathlib import Path
import shlex
import shutil
import subprocess


def run_synthseg(
    input_image,
    output_dir,
    synthseg_cmd="mri_synthseg",
    overwrite=False,
):
    input_image = Path(input_image)
    output_dir = Path(output_dir)

    if not input_image.exists():
        return None, "missing"

    output_file = output_dir / "synthseg.nii.gz"
    if output_file.exists() and not overwrite:
        return output_file, "skipped"

    if shutil.which(synthseg_cmd) is None:
        return None, "missing_command"

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        synthseg_cmd,
        "--i", str(input_image),
        "--o", str(output_file),
        "--cpu",
    ]

    command_log = output_dir / "synthseg_command.txt"
    stdout_log = output_dir / "synthseg_stdout.log"
    stderr_log = output_dir / "synthseg_stderr.log"
    command_log.write_text(shlex.join(cmd) + "\n")

    with stdout_log.open("w") as stdout_f, stderr_log.open("w") as stderr_f:
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=stdout_f,
                stderr=stderr_f,
            )
        except Exception as exc:
            stderr_f.write(f"{type(exc).__name__}: {exc}\n")
            return None, "failed"

    if result.returncode != 0:
        return None, "failed"

    if output_file.exists():
        return output_file, "done"

    return None, "failed"
