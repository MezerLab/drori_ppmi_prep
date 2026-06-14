from __future__ import annotations
import argparse
import json
from pathlib import Path

from drori_ppmi_prep.analysis.dataset_builder import build_analysis_dataset_from_metadata
from drori_ppmi_prep.clinical.cohort_tables import build_cohort_clinical_tables
from drori_ppmi_prep.conversion.dicom_to_nifti import convert_ppmi_dicoms_to_nifti
from drori_ppmi_prep.metadata.builder import build_ppmi_metadata_csv
from drori_ppmi_prep.metadata.dicom_enrichment import enrich_metadata_with_dicom_info


def write_ppmi_config(
    ppmi_root,
    idaSearch_dir,
    output_root,
    metadata_csv,
    nifti_root,
    analysis_root,
    group_analysis_root,
    freesurfer_root,
    cohort_tables_root=None,
    study_tables_root=None,
):
    config = {
        "ppmi_root": str(Path(ppmi_root).resolve()),
        "idaSearch_dir": str(Path(idaSearch_dir).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "metadata_csv": str(Path(metadata_csv).resolve()),
        "nifti_root": str(Path(nifti_root).resolve()),
        "analysis_root": str(Path(analysis_root).resolve()),
        "group_analysis_root": str(Path(group_analysis_root).resolve()),
        "freesurfer_root": str(Path(freesurfer_root).resolve()),
    }
    if cohort_tables_root is not None:
        config["cohort_tables_root"] = str(Path(cohort_tables_root).resolve())
    if study_tables_root is not None:
        config["study_tables_root"] = str(Path(study_tables_root).resolve())

    config_path = Path(output_root) / "ppmi_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def run_build_infrastructure(
    ppmi_root: str,
    idaSearch_dir: str,
    output_root: str,
    file_pattern: str = "*.csv",
    dcm2niix_cmd: str = "dcm2niix",
    force: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    study_tables_root: str | None = None,
    skip_cohort_tables: bool = False,
    force_cohort_tables: bool = False,
):
    ppmi_root = Path(ppmi_root)
    idaSearch_dir = Path(idaSearch_dir)
    output_root = Path(output_root)

    metadata_csv = output_root / "ppmi_metadata.csv"
    nifti_root = output_root / "PPMI_nifti"
    analysis_root = output_root / "PPMI_analysis"

    group_analysis_root = output_root / "group_analysis"
    freesurfer_root = group_analysis_root / "FreeSurfer"
    cohort_tables_root = output_root / "cohort_tables"

    print("-" * 70)
    print("Building dataset infrastructure...")
    print("-" * 70)

    output_root.mkdir(parents=True, exist_ok=True)

    total_steps = 5

    print(f"  (1/{total_steps}): Building metadata CSV...")
    build_ppmi_metadata_csv(
        input_dir=idaSearch_dir,
        output_csv=metadata_csv,
        file_pattern=file_pattern,
    )

    print("         Enriching metadata from DICOM headers...")
    enrich_metadata_with_dicom_info(
        metadata_csv=metadata_csv,
        ppmi_root=ppmi_root,
        output_csv=metadata_csv,
        parallel=parallel,
        max_workers=max_workers,
    )

    print(f"  (2/{total_steps}): Converting DICOMs to NIfTI...")
    convert_ppmi_dicoms_to_nifti(
        input_root=ppmi_root,
        output_root=nifti_root,
        dcm2niix_path=dcm2niix_cmd,
        overwrite=force,
        parallel=parallel,
        max_workers=max_workers,
    )

    print(f"  (3/{total_steps}): Building analysis dataset...")
    build_analysis_dataset_from_metadata(
        metadata_csv=metadata_csv,
        nifti_root=nifti_root,
        output_root=analysis_root,
        overwrite=force,
    )

    print(f"  (4/{total_steps}): Building cohort clinical tables...")
    if skip_cohort_tables:
        print("         SKIPPED: disabled by --skip-cohort-tables")
    elif study_tables_root is None:
        print("         SKIPPED: no --study-tables-root provided")
    else:
        result = build_cohort_clinical_tables(
            metadata_csv=metadata_csv,
            study_tables_root=study_tables_root,
            output_dir=cohort_tables_root,
            overwrite=force or force_cohort_tables,
        )
        print(f"         Tables written/found: {len(result['tables'])}")
        for name, reason in result["skipped"].items():
            print(f"         SKIPPED {name}: {reason}")
        for name, (_, status) in result["metrics"].items():
            print(f"         METRIC {name}: {status}")

    print(f"  (5/{total_steps}): Writing config file...")
    config_path = write_ppmi_config(
        ppmi_root=ppmi_root,
        idaSearch_dir=idaSearch_dir,
        output_root=output_root,
        metadata_csv=metadata_csv,
        nifti_root=nifti_root,
        analysis_root=analysis_root,
        group_analysis_root=group_analysis_root,
        freesurfer_root=freesurfer_root,
        cohort_tables_root=cohort_tables_root,
        study_tables_root=study_tables_root,
    )

    print("Infrastructure build finished.")
    print(f"  Metadata CSV: {metadata_csv}")
    print(f"  NIfTI root: {nifti_root}")
    print(f"  Analysis root: {analysis_root}")
    print(f"  Cohort tables root: {cohort_tables_root}")
    print(f"  Config file: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build shared PPMI infrastructure: metadata, NIfTI conversion, analysis dataset, and config."
    )
    parser.add_argument("ppmi_root")
    parser.add_argument("idaSearch_dir")
    parser.add_argument("output_root")
    parser.add_argument("--file-pattern", default="*.csv")
    parser.add_argument("--dcm2niix-cmd", default="dcm2niix")
    parser.add_argument(
        "--study-tables-root",
        default=None,
        help="Root directory containing downloaded PPMI study tables.",
    )
    parser.add_argument("--skip-cohort-tables", action="store_true")
    parser.add_argument("--force-cohort-tables", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run dicom conversion in parallel.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel workers. Defaults to the number of CPU cores.",
    )

    args = parser.parse_args()

    run_build_infrastructure(
        ppmi_root=args.ppmi_root,
        idaSearch_dir=args.idaSearch_dir,
        output_root=args.output_root,
        file_pattern=args.file_pattern,
        dcm2niix_cmd=args.dcm2niix_cmd,
        force=args.force,
        parallel=args.parallel,
        max_workers=args.max_workers,
        study_tables_root=args.study_tables_root,
        skip_cohort_tables=args.skip_cohort_tables,
        force_cohort_tables=args.force_cohort_tables,
    )


if __name__ == "__main__":
    main()
