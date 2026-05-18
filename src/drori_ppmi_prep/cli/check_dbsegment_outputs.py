import argparse
import csv
import json
from pathlib import Path


DBSEGMENT_INPUT = Path("t1_space/segmentation/synthstrip/T1_brainmask.nii.gz")
DBSEGMENT_OUTPUT = Path("t1_space/segmentation/dbsegment/T1.nii.gz")


def resolve_analysis_root(path):
    path = Path(path)

    config_path = path / "ppmi_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return Path(config["analysis_root"])

    analysis_root = path / "PPMI_analysis"
    if analysis_root.exists():
        return analysis_root

    return path


def iter_session_dirs(analysis_root):
    for subject_dir in sorted(analysis_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        for session_dir in sorted(subject_dir.iterdir()):
            if session_dir.is_dir():
                yield subject_dir.name, session_dir.name, session_dir


def check_dbsegment_outputs(analysis_root):
    analysis_root = Path(analysis_root)

    if not analysis_root.exists():
        raise FileNotFoundError(f"Analysis root not found: {analysis_root}")

    rows = []

    for subject_id, session_id, session_dir in iter_session_dirs(analysis_root):
        input_path = session_dir / DBSEGMENT_INPUT
        output_path = session_dir / DBSEGMENT_OUTPUT

        input_exists = input_path.exists()
        output_exists = output_path.exists()

        if not input_exists:
            status = "not_applicable"
        elif output_exists:
            status = "done"
        else:
            status = "missing_output"

        rows.append({
            "subject_id": subject_id,
            "session_id": session_id,
            "session_dir": str(session_dir),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_exists": input_exists,
            "output_exists": output_exists,
            "status": status,
        })

    return rows


def write_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subject_id",
        "session_id",
        "session_dir",
        "input_path",
        "output_path",
        "input_exists",
        "output_exists",
        "status",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check DBSegment completeness across analysis sessions. A session is "
            "applicable when t1_space/segmentation/synthstrip/T1_brainmask.nii.gz exists."
        )
    )
    parser.add_argument(
        "path",
        help="OUTPUT_ROOT, a directory containing PPMI_analysis, or PPMI_analysis itself.",
    )
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help="Print subject/session rows for applicable sessions missing DBSegment output.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path for the full per-session status table.",
    )

    args = parser.parse_args()

    analysis_root = resolve_analysis_root(args.path)
    rows = check_dbsegment_outputs(analysis_root)

    total_sessions = len(rows)
    applicable = [row for row in rows if row["input_exists"]]
    done = [row for row in applicable if row["output_exists"]]
    missing = [row for row in applicable if not row["output_exists"]]
    not_applicable = [row for row in rows if not row["input_exists"]]

    print(f"Analysis root              : {analysis_root}")
    print(f"Total sessions             : {total_sessions}")
    print(f"DBSegment applicable       : {len(applicable)}")
    print(f"DBSegment outputs present  : {len(done)}")
    print(f"DBSegment outputs missing  : {len(missing)}")
    print(f"Not applicable             : {len(not_applicable)}")

    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"CSV report                 : {args.csv}")

    if args.list_missing and missing:
        print()
        print("Missing DBSegment outputs:")
        for row in missing:
            print(f"{row['subject_id']} / {row['session_id']}")
            print(f"  session: {row['session_dir']}")
            print(f"  input  : {row['input_path']}")
            print(f"  output : {row['output_path']}")


if __name__ == "__main__":
    main()
