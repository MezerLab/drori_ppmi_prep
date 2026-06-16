import argparse
import json
from pathlib import Path

from drori_ppmi_prep.clinical.cohort_tables import (
    CLINICAL_TABLES_BY_NAME,
    build_cohort_clinical_tables,
)


def resolve_metadata_csv(output_root, metadata_csv):
    if metadata_csv is not None:
        return Path(metadata_csv)
    output_root = Path(output_root)
    config_path = output_root / "ppmi_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return Path(config["metadata_csv"])
    return output_root / "ppmi_metadata.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Build cohort-specific clinical tables from PPMI study tables."
    )
    parser.add_argument("output_root")
    parser.add_argument("study_tables_root")
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Metadata CSV to use. Defaults to OUTPUT_ROOT/ppmi_config.json or ppmi_metadata.csv.",
    )
    parser.add_argument(
        "--table",
        choices=tuple(CLINICAL_TABLES_BY_NAME),
        action="append",
        help="Generate only this clinical table. Can be repeated. Defaults to all known tables.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-calculated-metrics",
        action="store_true",
        help="Do not generate derived clinical metric tables.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_dir = output_root / "cohort_tables"
    metadata_csv = resolve_metadata_csv(output_root, args.metadata_csv)

    result = build_cohort_clinical_tables(
        metadata_csv=metadata_csv,
        study_tables_root=args.study_tables_root,
        output_dir=output_dir,
        table_names=args.table,
        overwrite=args.overwrite,
        calculate_metrics=not args.skip_calculated_metrics,
    )

    print(f"Cohort tables directory: {output_dir}")
    for name, path in result["tables"].items():
        print(f"  TABLE {name}: {path}")
    for name, reason in result["skipped"].items():
        print(f"  SKIPPED {name}: {reason}")
    qa_path, qa_status = result["imaging_qa"]
    print(f"  IMAGING_QA: {qa_status}: {qa_path}")
    for name, (path, status) in result["metrics"].items():
        print(f"  METRIC {name}: {status}: {path}")


if __name__ == "__main__":
    main()
