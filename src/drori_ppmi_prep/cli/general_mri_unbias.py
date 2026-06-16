import argparse

from mri_unbias.io import unbias_nifti


def main():
    parser = argparse.ArgumentParser(
        description="Run polynomial MRI bias correction for one image."
    )
    parser.add_argument("--image", required=True, help="Input image.")
    parser.add_argument("--mask", required=True, help="Homogeneous-region mask.")
    parser.add_argument("--corrected", required=True, help="Output corrected image.")
    parser.add_argument("--bias-field", required=True, help="Output bias-field image.")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--brain-mask", default=None, help="Optional brain mask.")

    args = parser.parse_args()
    unbias_nifti(
        image_path=args.image,
        mask_path=args.mask,
        corrected_path=args.corrected,
        bias_field_path=args.bias_field,
        degree=args.degree,
        brain_mask_path=args.brain_mask,
    )

    print(f"Corrected image: {args.corrected}")
    print(f"Bias field     : {args.bias_field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
