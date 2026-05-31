import shutil
import shlex
import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


GP_SN_LABEL_GROUPS = {
    4: (4, 6),
    5: (5, 7),
    18: (18, 20),
    19: (19, 21),
}


def create_gp_sn_segmentation(dbsegment_file, output_file, overwrite=False):
    dbsegment_file = Path(dbsegment_file)
    output_file = Path(output_file)

    if not dbsegment_file.exists():
        return None, "missing"

    if output_file.exists() and not overwrite:
        return output_file, "skipped"

    source_img = nib.load(str(dbsegment_file))
    source_data = np.rint(source_img.get_fdata()).astype(np.int32)
    derivative_data = np.zeros(source_data.shape, dtype=np.uint8)

    for output_label, source_labels in GP_SN_LABEL_GROUPS.items():
        derivative_data[np.isin(source_data, source_labels)] = output_label

    output_file.parent.mkdir(parents=True, exist_ok=True)
    derivative_img = nib.Nifti1Image(
        derivative_data,
        source_img.affine,
        source_img.header,
    )
    derivative_img.set_data_dtype(np.uint8)
    nib.save(derivative_img, str(output_file))

    return output_file, "done"


def remove_dbsegment_logs(output_dir):
    for log_file in [
        output_dir / "dbsegment_command.txt",
        output_dir / "dbsegment_stdout.log",
        output_dir / "dbsegment_stderr.log",
    ]:
        if log_file.exists():
            log_file.unlink()


def run_dbsegment(
    input_image,
    output_dir,
    model_path=None,
    dbsegment_cmd="DBSegment",
    overwrite=False,
    use_cuda=True,
):
    input_image = Path(input_image)
    output_dir = Path(output_dir)

    if not input_image.exists():
        return None, "missing"

    output_file = output_dir / "T1.nii.gz"
    derivative_file = output_dir / "derivatives" / "GP_SN_seg.nii.gz"

    if output_file.exists() and not overwrite:
        _, derivative_status = create_gp_sn_segmentation(output_file, derivative_file)
        if derivative_status not in {"done", "skipped"}:
            return None, "failed"
        return output_file, "skipped"

    if shutil.which(dbsegment_cmd) is None:
        return None, "missing_command"

    output_dir.mkdir(parents=True, exist_ok=True)
    command_log = output_dir / "dbsegment_command.txt"
    stdout_log = output_dir / "dbsegment_stdout.log"
    stderr_log = output_dir / "dbsegment_stderr.log"

    with tempfile.TemporaryDirectory(prefix="dbsegment_") as tmpdir:
        tmpdir = Path(tmpdir)
        tmpfile = tmpdir / "T1.nii.gz"

        shutil.copy2(input_image, tmpfile)

        cmd = [
            dbsegment_cmd,
            "-i",
            str(tmpdir),
            "-o",
            str(output_dir),
        ]

        if model_path is not None:
            cmd.extend(["-mp", str(Path(model_path))])

        env = os.environ.copy()
        if not use_cuda:
            env["CUDA_VISIBLE_DEVICES"] = ""
            existing_warnings = env.get("PYTHONWARNINGS", "")
            cuda_warning_filter = "ignore:CUDA initialization:UserWarning"
            env["PYTHONWARNINGS"] = (
                f"{existing_warnings},{cuda_warning_filter}"
                if existing_warnings
                else cuda_warning_filter
            )
            cmd.extend(["--all_in_gpu", "False"])

        env_prefix = (
            "CUDA_VISIBLE_DEVICES='' "
            f"PYTHONWARNINGS={shlex.quote(env['PYTHONWARNINGS'])} "
            if not use_cuda
            else ""
        )
        command_log.write_text(env_prefix + shlex.join(cmd) + "\n")

        with stdout_log.open("w") as stdout_f, stderr_log.open("w") as stderr_f:
            try:
                result = subprocess.run(
                    cmd,
                    text=True,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    env=env,
                )
            except Exception as e:
                stderr_f.write(f"{type(e).__name__}: {e}\n")
                return None, "failed"

        if output_file.exists():
            _, derivative_status = create_gp_sn_segmentation(
                output_file,
                derivative_file,
                overwrite=overwrite,
            )
            if derivative_status not in {"done", "skipped"}:
                return None, "failed"
            remove_dbsegment_logs(output_dir)
            return output_file, "done"

        if result.returncode != 0:
            return None, "failed"

    return output_file, "failed"
