import argparse
from pathlib import Path

from drori_ppmi_prep.segmentation.freesurfer import (
    export_all_freesurfer_mgz_to_orig_space,
    run_freesurfer,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run FreeSurfer recon-all for one image and export FreeSurfer MRI "
            "outputs into a reference image space."
        )
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input T1 image for recon-all. Required unless --export-only is used.",
    )
    parser.add_argument("--subjects-dir", required=True, help="FreeSurfer SUBJECTS_DIR.")
    parser.add_argument("--subject-id", required=True, help="FreeSurfer subject ID.")
    parser.add_argument(
        "--reference-image",
        required=True,
        help="Reference image for mri_vol2vol export.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for exported reference-space NIfTI files.",
    )
    parser.add_argument("--freesurfer-cmd", default="recon-all")
    parser.add_argument("--mri-vol2vol-cmd", default="mri_vol2vol")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip recon-all and export an existing FreeSurfer subject.",
    )
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="Delete and restart an incomplete FreeSurfer subject directory.",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    subjects_dir = Path(args.subjects_dir)
    subject_dir = subjects_dir / args.subject_id
    freesurfer_status = "skipped"

    if not args.export_only:
        if args.input is None:
            raise ValueError("--input is required unless --export-only is used.")
        subject_dir, freesurfer_status = run_freesurfer(
            input_image=args.input,
            subjects_dir=subjects_dir,
            subject_id=args.subject_id,
            recon_all_cmd=args.freesurfer_cmd,
            overwrite=args.overwrite,
            restart_incomplete=args.restart_incomplete,
        )
        if freesurfer_status not in {"done", "skipped"}:
            print(f"FreeSurfer status: {freesurfer_status}")
            return 1

    mri_dir = subject_dir / "mri"
    exported, export_status = export_all_freesurfer_mgz_to_orig_space(
        freesurfer_mri_dir=mri_dir,
        reference_t1=args.reference_image,
        output_dir=args.output_dir,
        mri_vol2vol_cmd=args.mri_vol2vol_cmd,
        overwrite=args.overwrite,
    )

    print(f"FreeSurfer status: {freesurfer_status}")
    print(f"Export status    : {export_status}")
    print(f"Subject dir      : {subject_dir}")
    print(f"Output dir       : {args.output_dir}")
    print(f"Exported files   : {len(exported)}")

    return 0 if export_status in {"done", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
