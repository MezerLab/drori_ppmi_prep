import argparse

from drori_ppmi_prep.analysis.roi_stats import (
    IMAGE_PATHS,
    ROI_STATS,
    SEGMENTATIONS,
    run_roi_stats,
)


def available_images():
    return tuple(
        dict.fromkeys(
            image_name
            for paths in IMAGE_PATHS.values()
            for image_name in paths
        )
    )


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
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N completed sessions. Default: 50.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print progress while processing sessions.",
    )
    parser.add_argument(
        "--image",
        choices=available_images(),
        action="append",
        help=(
            "Generate intensity tables only for this image name. Can be repeated. "
            "Defaults to t1, t2, and pd."
        ),
    )
    parser.add_argument(
        "--stat",
        choices=ROI_STATS,
        action="append",
        help=(
            "Generate only this statistic. Can be repeated. Choices include "
            "median, mean, mad, std, and volume. Defaults to all stats."
        ),
    )
    parser.add_argument(
        "--segmentation",
        choices=tuple(SEGMENTATIONS),
        action="append",
        help=(
            "Generate ROI_stats only for this segmentation. Can be repeated. "
            "Defaults to all segmentations."
        ),
    )
    args = parser.parse_args()

    output_dir, status = run_roi_stats(
        output_root=args.output_root,
        freesurfer_lut=args.freesurfer_lut,
        overwrite=args.overwrite,
        parallel=args.parallel,
        max_workers=args.max_workers,
        selected_segmentations=args.segmentation,
        selected_images=args.image,
        selected_stats=args.stat,
        progress=not args.quiet,
        progress_interval=args.progress_interval,
    )
    if status == "skipped":
        print(f"SKIPPED: already done: {output_dir}")
    else:
        print(f"DONE: {output_dir}")


if __name__ == "__main__":
    main()
