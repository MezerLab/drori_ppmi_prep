import argparse
from pathlib import Path

from drori_ppmi_prep.segmentation.massp import (
    AHEAD_TEMPLATE_ARTICLE_ID,
    AHEAD_TEMPLATE_FILENAME,
    MASSP_ATLAS_ARTICLE_ID,
    MASSP_ATLAS_FILENAME,
    resolve_massp_resource,
)


def main():
    parser = argparse.ArgumentParser(
        description="Download the AHEAD template and MASSP 2021 older-adults atlas."
    )
    parser.add_argument(
        "output_root",
        help="Pipeline output root where group_analysis/atlases/massp2021 will be created.",
    )

    args = parser.parse_args()
    cache_dir = Path(args.output_root) / "group_analysis" / "atlases" / "massp2021"

    template = resolve_massp_resource(
        None,
        cache_dir,
        AHEAD_TEMPLATE_ARTICLE_ID,
        AHEAD_TEMPLATE_FILENAME,
        allow_download=True,
    )
    atlas = resolve_massp_resource(
        None,
        cache_dir,
        MASSP_ATLAS_ARTICLE_ID,
        MASSP_ATLAS_FILENAME,
        allow_download=True,
    )

    if template is None or atlas is None:
        raise RuntimeError("Failed to download one or more MASSP resources from Figshare.")

    print(f"AHEAD template: {template}")
    print(f"MASSP atlas   : {atlas}")


if __name__ == "__main__":
    main()
