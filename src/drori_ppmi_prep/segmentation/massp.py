from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


AHEAD_TEMPLATE_ARTICLE_ID = "12301106"
MASSP_ATLAS_ARTICLE_ID = "19646346"
AHEAD_TEMPLATE_FILENAME = "ahead_med_qr1.nii.gz"
MASSP_ATLAS_FILENAME = "massp2021-parcellation_decade-61to80.nii.gz"

README_TEXT = """This segmentation is based on nonlinear registration of the AHEAD average R1 image and MASSP 2021 Older Adults atlas to the subject T1 reference space.
The AHEAD R1 template is registered to the subject brainmasked T1 image with ANTs affine + SyN registration.
The MASSP parcellation is transformed with ANTs using nearest-neighbor interpolation.
Atlas source: https://doi.org/10.21942/uva.19646346
Template source: https://doi.org/10.21942/uva.12301106
License: CC BY 4.0. Please cite the source datasets when using this output.
"""


def _download_file(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with urlopen(url) as response, tmp_path.open("wb") as f:
            shutil.copyfileobj(response, f)
    except URLError:
        if tmp_path.exists():
            tmp_path.unlink()
        return None

    tmp_path.replace(output_path)
    return output_path


def _figshare_file_download_url(article_id: str, filename: str):
    try:
        with urlopen(f"https://api.figshare.com/v2/articles/{article_id}/files") as response:
            files = json.loads(response.read().decode("utf-8"))
    except URLError:
        return None

    for file_info in files:
        if file_info.get("name") == filename:
            return file_info.get("download_url")

    return None


def resolve_massp_resource(
    path: str | Path | None,
    cache_dir: str | Path,
    article_id: str,
    filename: str,
    allow_download: bool = True,
):
    if path is not None:
        path = Path(path)
        return path if path.exists() else None

    cache_dir = Path(cache_dir)
    cached_path = cache_dir / filename
    if cached_path.exists():
        return cached_path

    if not allow_download:
        return None

    download_url = _figshare_file_download_url(article_id, filename)
    if download_url is None:
        return None

    return _download_file(download_url, cached_path)


def remove_massp_logs(output_dir: Path):
    for log_file in [
        output_dir / "massp_registration_command.txt",
        output_dir / "massp_apply_command.txt",
        output_dir / "massp_registration_stdout.log",
        output_dir / "massp_registration_stderr.log",
        output_dir / "massp_apply_stdout.log",
        output_dir / "massp_apply_stderr.log",
    ]:
        if log_file.exists():
            log_file.unlink()


def run_massp_atlas_segmentation(
    target_image,
    output_dir,
    atlas_path,
    template_path,
    ants_registration_cmd="antsRegistration",
    ants_apply_cmd="antsApplyTransforms",
    overwrite=False,
):
    target_image = Path(target_image)
    output_dir = Path(output_dir)
    atlas_path = Path(atlas_path) if atlas_path is not None else None
    template_path = Path(template_path) if template_path is not None else None

    if not target_image.exists() or atlas_path is None or template_path is None:
        return None, "missing"

    if not atlas_path.exists() or not template_path.exists():
        return None, "missing"

    warped_template = output_dir / "ahead_med_qr1_2ref.nii.gz"
    warped_atlas = output_dir / "massp2021-parcellation_decade-61to80_2ref.nii.gz"
    readme_file = output_dir / "README.txt"
    transform_prefix = str(output_dir / "ahead2sub_")
    affine_transform = output_dir / "ahead2sub_0GenericAffine.mat"
    warp_transform = output_dir / "ahead2sub_1Warp.nii.gz"

    expected_outputs = [
        warped_template,
        warped_atlas,
        affine_transform,
        warp_transform,
        readme_file,
    ]
    if all(path.exists() for path in expected_outputs) and not overwrite:
        return warped_atlas, "skipped"

    if shutil.which(ants_registration_cmd) is None or shutil.which(ants_apply_cmd) is None:
        return None, "missing_command"

    output_dir.mkdir(parents=True, exist_ok=True)

    registration_cmd = [
        ants_registration_cmd,
        "--dimensionality", "3",
        "--float", "1",
        "--output", f"[{transform_prefix},{warped_template}]",
        "--interpolation", "BSpline",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "0",
        "--initial-moving-transform", f"[{target_image},{template_path},1]",
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{target_image},{template_path},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--transform", "Affine[0.1]",
        "--metric", f"MI[{target_image},{template_path},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--transform", "SyN[0.1,3,0]",
        "--metric", f"CC[{target_image},{template_path},1,4]",
        "--convergence", "[100x70x50x20,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
    ]
    apply_cmd = [
        ants_apply_cmd,
        "-d", "3",
        "-i", str(atlas_path),
        "-r", str(target_image),
        "-o", str(warped_atlas),
        "-n", "NearestNeighbor",
        "-t", str(warp_transform),
        "-t", str(affine_transform),
    ]

    (output_dir / "massp_registration_command.txt").write_text(
        shlex.join(registration_cmd) + "\n"
    )
    (output_dir / "massp_apply_command.txt").write_text(
        shlex.join(apply_cmd) + "\n"
    )

    with (output_dir / "massp_registration_stdout.log").open("w") as stdout_f:
        stderr_log = output_dir / "massp_registration_stderr.log"
        stderr_f = stderr_log.open("w")
        try:
            result = subprocess.run(
                registration_cmd,
                text=True,
                stdout=stdout_f,
                stderr=stderr_f,
            )
        finally:
            stderr_f.close()

    if result.returncode != 0 or not affine_transform.exists() or not warp_transform.exists():
        return None, "failed"

    with (output_dir / "massp_apply_stdout.log").open("w") as stdout_f:
        stderr_log = output_dir / "massp_apply_stderr.log"
        stderr_f = stderr_log.open("w")
        try:
            result = subprocess.run(
                apply_cmd,
                text=True,
                stdout=stdout_f,
                stderr=stderr_f,
            )
        finally:
            stderr_f.close()

    if readme_file.exists() and overwrite:
        readme_file.unlink()
    if not readme_file.exists():
        readme_file.write_text(README_TEXT)

    if result.returncode == 0 and all(path.exists() for path in expected_outputs):
        remove_massp_logs(output_dir)
        return warped_atlas, "done"

    return None, "failed"
