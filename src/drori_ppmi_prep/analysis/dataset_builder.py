from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


def is_gzip_file(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def build_analysis_dataset_from_metadata(
    metadata_csv: str | Path,
    nifti_root: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Build an analysis dataset from metadata and a NIfTI root directory.

    For each row in the metadata table, create:
        output_root/SubjectID/SessionID/

    Inside that directory, create symlinks:
        T1.nii.gz
        T2.nii.gz
        PD.nii.gz

    The source NIfTI files are expected under a structure derived from the original
    converted dataset, where the final file name is based on the image ID:
        nifti_root/SUBJECT_ID/SEQUENCE_NAME/SESSION_ID/IIMAGE_ID.nii.gz
        or
        nifti_root/SUBJECT_ID/SEQUENCE_NAME/SESSION_ID/IMAGE_ID.nii.gz

    The unsuffixed T1/T2/PD columns are expected to contain the selected
    analysis image IDs. Raw candidates can be retained in T1_1/T1_2/etc. by the
    metadata builder, but selection is not repeated here.

    Parameters
    ----------
    metadata_csv : str or Path
        Path to the metadata CSV.
    nifti_root : str or Path
        Root directory of the converted NIfTI dataset.
    output_root : str or Path
        Root directory where the analysis dataset will be created.
    overwrite : bool
        If True, replace existing symlinks.

    Returns
    -------
    pd.DataFrame
        A summary table with one row per metadata row, including the selected image IDs.
    """
    metadata_csv = Path(metadata_csv)
    nifti_root = Path(nifti_root)
    output_root = Path(output_root)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not nifti_root.exists():
        raise FileNotFoundError(f"NIfTI root not found: {nifti_root}")

    df = pd.read_csv(metadata_csv)

    required_cols = ["SubjectID", "SessionID"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")

    def is_missing_value(value: object) -> bool:
        if pd.isna(value):
            return True
        s = str(value).strip()
        return s == "" or s == "0" or s.lower() == "nan"

    def normalize_numeric_id(value: object) -> str | None:
        if is_missing_value(value):
            return None
        s = str(value).strip()
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
        return s

    def find_nifti_for_image(subject_id: object, session_id: object, image_id: str | None) -> Path | None:
        if image_id is None:
            return None

        subject_str = normalize_numeric_id(subject_id)
        session_str = str(session_id).strip()

        if subject_str is None or session_str == "":
            return None

        subject_dir = nifti_root / subject_str
        if not subject_dir.exists():
            return None

        candidates = [
            f"I{image_id}.nii.gz",
            f"{image_id}.nii.gz",
            f"I{image_id}_e1.nii.gz",
            f"{image_id}_e1.nii.gz",
            f"I{image_id}_e2.nii.gz",
            f"{image_id}_e2.nii.gz",
        ]

        # Expected structure: SUBJECT/SEQUENCE/SESSION/file
        for filename in candidates:
            matches = list(subject_dir.glob(f"*/{session_str}/{filename}"))
            valid_matches = [path for path in matches if is_gzip_file(path)]
            if valid_matches:
                return valid_matches[0]

        return None

    def safe_symlink(src: Path, dst: Path, overwrite_link: bool = False) -> None:
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink():
                current_target = dst.resolve()
                expected_target = src.resolve()
                if current_target == expected_target:
                    return
                dst.unlink()
                os.symlink(src, dst)
                return

            if not overwrite_link:
                return
            dst.unlink()
        os.symlink(src, dst)

    selections = []

    for _, row in df.iterrows():
        subject_id = normalize_numeric_id(row["SubjectID"])
        session_id = str(row["SessionID"]).strip() if pd.notna(row["SessionID"]) else ""

        if subject_id is None or session_id == "":
            selections.append({
                "SubjectID": row.get("SubjectID", ""),
                "SessionID": row.get("SessionID", ""),
                "SelectedT1": "",
                "SelectedT2": "",
                "SelectedPD": "",
                "AnalysisDir": "",
            })
            continue

        analysis_dir = output_root / subject_id / session_id
        analysis_dir.mkdir(parents=True, exist_ok=True)

        selected = {}

        for weighting in ["T1", "T2", "PD"]:
            image_id = normalize_numeric_id(row.get(weighting))
            description = ""
            desc_col = f"{weighting}_Description"
            if desc_col in row.index and pd.notna(row[desc_col]):
                description = str(row[desc_col])
            nifti_path = find_nifti_for_image(subject_id, session_id, image_id)

            selected[weighting] = image_id if image_id is not None else ""
            selected[f"{weighting}_Description"] = description
            selected[f"{weighting}_Path"] = str(nifti_path) if nifti_path is not None else ""

            if nifti_path is not None:
                safe_symlink(nifti_path, analysis_dir / f"{weighting}.nii.gz", overwrite_link=overwrite)

        selections.append({
            "SubjectID": subject_id,
            "SessionID": session_id,
            "SelectedT1": selected["T1"],
            "SelectedT2": selected["T2"],
            "SelectedPD": selected["PD"],
            "AnalysisDir": str(analysis_dir),
        })

    summary_df = pd.DataFrame(selections)
    return summary_df
