import argparse

from drori_ppmi_prep.analysis.roi_stats import run_roi_stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate group-level ROI statistics tables from T1-space segmentations."
    )
    parser.add_argument("output_root")
    parser.add_argument(
        "--freesurfer-lut",
        default=None,
        help="Optional path to FreeSurferColorLUT.txt. Defaults to FREESURFER_HOME.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    output_dir, status = run_roi_stats(
        output_root=args.output_root,
        freesurfer_lut=args.freesurfer_lut,
        overwrite=args.overwrite,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )
    if status == "skipped":
        print(f"SKIPPED: already done: {output_dir}")
    else:
        print(f"DONE: {output_dir}")


if __name__ == "__main__":
    main()
