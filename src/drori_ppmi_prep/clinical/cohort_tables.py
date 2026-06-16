from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ClinicalTableSpec:
    name: str
    relative_path: str
    date_column: str = "INFODT"
    unlimited_date_diff: bool = False


CLINICAL_TABLES = (
    ClinicalTableSpec(
        "moca",
        "Non-motor/Neuropsychological/Montreal_Cognitive_Assessment__MoCA_.csv",
    ),
    ClinicalTableSpec(
        "updrs3",
        "Motor/MDS-UPDRS/MDS_UPDRS_Part_III.csv",
    ),
    ClinicalTableSpec(
        "rbd",
        "Non-motor/Sleep_Disorder/REM_Sleep_Behavior_Disorder_Questionnaire.csv",
    ),
    ClinicalTableSpec(
        "scopa",
        "Non-motor/Autonomic/SCOPA-AUT.csv",
    ),
    ClinicalTableSpec(
        "olfactory",
        "Non-motor/Olfactory/University_of_Pennsylvania_Smell_Identification_Test__UPSIT_.csv",
    ),
    ClinicalTableSpec(
        "pd_diagnosis",
        "Medical_history/PD_Diagnosis_History.csv",
    ),
    ClinicalTableSpec(
        "demographics",
        "Subject_Characteristics/Demographics.csv",
    ),
    ClinicalTableSpec(
        "gba1_lrrk2",
        "Subject_Characteristics/Genetic_status/Consensus_APOE_SNPs_and GBA_and_LRRK2_Coding_Variant_Summary.csv",
    ),
    ClinicalTableSpec(
        "genetic_status",
        "Subject_Characteristics/Genetic_status/Participant_Genetic_Status_for_Selected_PD-Associated_Variants.csv",
    ),
    ClinicalTableSpec(
        "family_history",
        "Subject_Characteristics/Family_History.csv",
        unlimited_date_diff=True,
    ),
    ClinicalTableSpec(
        "datscan",
        "Imaging/DaTSCAN/DaTScan_Analysis.csv",
        date_column="DATSCAN_DATE",
    ),
    ClinicalTableSpec(
        "datscan_xi",
        "Imaging/DaTSCAN_XI/Xing_Core_Lab_-_Quant_SBR_15Dec2025.csv",
        date_column="DATSCAN_DATE",
    ),
    ClinicalTableSpec(
        "parkinsonism",
        "Medical_history/Features_of_Parkinsonism.csv",
    ),
    ClinicalTableSpec(
        "other_clinical_features",
        "Non-motor/Other_Clinical_Features.csv",
    ),
)

CLINICAL_TABLES_BY_NAME = {
    spec.name: spec
    for spec in CLINICAL_TABLES
}

BASE_COLUMNS = (
    "RowID",
    "SubjectID",
    "SessionID",
)

IMAGING_QA_COLUMNS = (
    "image_QA",
    "fslfirst_QA",
    "massp_QA",
    "abnormality_QA",
    "motion_QA",
)


def is_empty_qa_value(value):
    if pd.isna(value):
        return True
    value = str(value).strip()
    return value == "" or value.lower() == "nan"


def existing_imaging_qa_is_empty(path):
    path = Path(path)
    if not path.exists():
        return True

    table = pd.read_csv(path, dtype=str, low_memory=False, keep_default_na=True)
    value_columns = [column for column in table.columns if column not in BASE_COLUMNS]
    if not value_columns:
        return True

    return table.loc[:, value_columns].apply(lambda col: col.map(is_empty_qa_value)).all().all()


def normalize_subject_id(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return value


def is_missing_value(value):
    if pd.isna(value):
        return True
    value = str(value).strip()
    return value == "" or value.lower() == "nan"


def cohort_table_base(metadata):
    missing = [col for col in BASE_COLUMNS if col not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")
    return metadata.loc[:, BASE_COLUMNS].copy()


def resolve_study_table(study_tables_root, relative_path):
    study_tables_root = Path(study_tables_root)
    relative_path = Path(relative_path)
    exact_path = study_tables_root / relative_path
    if exact_path.exists():
        return exact_path

    expected_name = relative_path.name
    exact_name_matches = sorted(study_tables_root.rglob(expected_name))
    if exact_name_matches:
        return exact_name_matches[-1]

    parent = study_tables_root / relative_path.parent
    matches = []
    if parent.exists():
        matches = sorted(parent.glob(f"{relative_path.stem}*.csv"))
    if not matches:
        matches = sorted(study_tables_root.rglob(f"{relative_path.stem}*.csv"))
    if not matches:
        return None
    return matches[-1]


def parse_ppmi_month_year(values):
    parsed = pd.to_datetime(values, format="%m/%Y", errors="coerce")
    missing = parsed.isna() & pd.Series(values).notna()
    if missing.any():
        fallback = pd.to_datetime(pd.Series(values)[missing], format="%m/%d/%Y", errors="coerce")
        parsed.loc[missing] = fallback
    return parsed


def select_nearest_visit(study_rows, imaging_date, date_column, unlimited_date_diff=False):
    if date_column not in study_rows.columns:
        return study_rows.iloc[0:0]

    imaging_date = pd.to_datetime(imaging_date, errors="coerce")
    if pd.isna(imaging_date):
        return study_rows.iloc[0:0]
    imaging_date = imaging_date.replace(day=1)

    visit_dates = parse_ppmi_month_year(study_rows[date_column])
    diff = (visit_dates - imaging_date).abs().dt.days
    if diff.isna().all():
        return study_rows.iloc[0:0]

    max_diff = diff.max() + 1 if unlimited_date_diff else 370
    nearest = study_rows[(diff == diff.min()) & (diff <= max_diff)].copy()
    if len(nearest) > 1:
        nearest_dates = parse_ppmi_month_year(nearest[date_column])
        same_year = nearest_dates.dt.year == imaging_date.year
        if same_year.any():
            nearest = nearest[same_year]
    return nearest


def filter_special_cases(table_name, rows):
    if table_name == "updrs3" and "PAG_NAME" in rows.columns:
        rows = rows[rows["PAG_NAME"].isin(["NUPDRS3"])]
    if len(rows) > 1 and "EVENT_ID" in rows.columns:
        without_screening = rows[~rows["EVENT_ID"].isin(["SC"])]
        if not without_screening.empty:
            rows = without_screening
    return rows


def generate_cohort_table(metadata, study_table, table_name, date_column, unlimited_date_diff=False):
    if "SubjectID" not in metadata.columns:
        raise ValueError("Metadata CSV must contain SubjectID.")
    if "SessionID" not in metadata.columns:
        raise ValueError("Metadata CSV must contain SessionID.")
    if "PATNO" not in study_table.columns:
        raise ValueError(f"Study table '{table_name}' must contain PATNO.")

    output = cohort_table_base(metadata)
    study_table = study_table.copy()
    study_table["_PATNO_NORMALIZED"] = study_table["PATNO"].apply(normalize_subject_id)

    visit_sensitive = study_table["_PATNO_NORMALIZED"].duplicated().any()
    study_date_column = "StudyDate" if "StudyDate" in metadata.columns else None

    for row_index, metadata_row in metadata.iterrows():
        if is_missing_value(metadata_row["SessionID"]):
            continue
        subject_id = normalize_subject_id(metadata_row["SubjectID"])
        rows = study_table[study_table["_PATNO_NORMALIZED"] == subject_id].copy()
        if rows.empty:
            continue

        if visit_sensitive and study_date_column is not None:
            rows = select_nearest_visit(
                rows,
                metadata_row[study_date_column],
                date_column,
                unlimited_date_diff=unlimited_date_diff,
            )

        rows = filter_special_cases(table_name, rows)
        if rows.empty:
            continue
        if len(rows) > 1:
            analysis_dir = metadata_row.get("AnalysisDir", "")
            raise ValueError(
                f"Too many entries ({len(rows)}) for subject/session: {analysis_dir}"
            )

        values = rows.iloc[0].drop(labels=["_PATNO_NORMALIZED"], errors="ignore")
        for column in values.index:
            if column not in output.columns:
                output[column] = pd.NA
            value = values[column]
            output.at[row_index, column] = pd.NA if pd.isna(value) else value

    return output


def write_cohort_tables(
    metadata_csv,
    study_tables_root,
    output_dir,
    table_names=None,
    overwrite=False,
):
    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir)
    study_tables_root = Path(study_tables_root)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not study_tables_root.exists():
        raise FileNotFoundError(f"Study tables root not found: {study_tables_root}")

    selected_specs = list(CLINICAL_TABLES)
    if table_names is not None:
        unknown = sorted(set(table_names) - set(CLINICAL_TABLES_BY_NAME))
        if unknown:
            available = ", ".join(CLINICAL_TABLES_BY_NAME)
            raise ValueError(
                f"Unknown clinical table(s): {', '.join(unknown)}. "
                f"Available tables: {available}"
            )
        selected_specs = [CLINICAL_TABLES_BY_NAME[name] for name in dict.fromkeys(table_names)]

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_csv, low_memory=False, keep_default_na=True)
    outputs = {}
    skipped = {}

    for spec in selected_specs:
        output_path = output_dir / f"ppmi_cohort_{spec.name}.csv"
        if output_path.exists() and not overwrite:
            outputs[spec.name] = output_path
            continue

        study_table_path = resolve_study_table(study_tables_root, spec.relative_path)
        if study_table_path is None:
            skipped[spec.name] = "missing study table"
            continue

        study_table = pd.read_csv(study_table_path, low_memory=False, keep_default_na=True)
        try:
            cohort_table = generate_cohort_table(
                metadata,
                study_table,
                spec.name,
                spec.date_column,
                unlimited_date_diff=spec.unlimited_date_diff,
            )
        except ValueError as exc:
            skipped[spec.name] = str(exc)
            continue

        cohort_table.to_csv(output_path, index=False)
        outputs[spec.name] = output_path

    return outputs, skipped


def write_imaging_qa_table(metadata_csv, output_dir, overwrite=False):
    output_path = Path(output_dir) / "ppmi_cohort_imaging_QA.csv"
    if output_path.exists():
        if not existing_imaging_qa_is_empty(output_path):
            return output_path, "skipped_existing_values"
        if not overwrite:
            return output_path, "skipped"

    metadata = pd.read_csv(
        metadata_csv,
        dtype={"RowID": str, "SubjectID": str, "SessionID": str},
        low_memory=False,
        keep_default_na=True,
    )
    table = cohort_table_base(metadata)
    for column in IMAGING_QA_COLUMNS:
        table[column] = pd.NA

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return output_path, "done"


def _read_cohort_table(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False, keep_default_na=True)


def _base_columns(table):
    return [col for col in BASE_COLUMNS if col in table.columns]


def calculate_motor_metrics(cohort_tables_dir, output_dir, overwrite=False):
    input_path = Path(cohort_tables_dir) / "ppmi_cohort_updrs3.csv"
    output_path = Path(output_dir) / "ppmi_cohort_calc_motor.csv"
    if output_path.exists() and not overwrite:
        return output_path, "skipped"

    table = _read_cohort_table(input_path)
    if table is None:
        return output_path, "missing"

    required_groups = {
        "left": ["NP3RIGLU", "NP3RIGLL", "NP3FTAPL", "NP3HMOVL", "NP3PRSPL", "NP3TTAPL", "NP3LGAGL", "NP3PTRML", "NP3KTRML", "NP3RTALU", "NP3RTALL"],
        "right": ["NP3RIGRU", "NP3RIGRL", "NP3FTAPR", "NP3HMOVR", "NP3PRSPR", "NP3TTAPR", "NP3LGAGR", "NP3PTRMR", "NP3KTRMR", "NP3RTARU", "NP3RTARL"],
        "bilateral": ["NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3RTALJ", "NP3RTCON"],
        "tremor_left": ["NP3PTRML", "NP3KTRML", "NP3RTALU", "NP3RTALL"],
        "tremor_right": ["NP3PTRMR", "NP3KTRMR", "NP3RTARU", "NP3RTARL"],
    }
    all_required = {col for cols in required_groups.values() for col in cols}
    missing = sorted(all_required - set(table.columns))
    if missing:
        return output_path, f"missing columns: {', '.join(missing)}"

    output = table.loc[:, _base_columns(table)].copy()
    output["UPDRS3_L"] = table[required_groups["left"]].sum(axis=1, min_count=1)
    output["UPDRS3_R"] = table[required_groups["right"]].sum(axis=1, min_count=1)
    output["UPDRS3_Asymm"] = output["UPDRS3_L"] - output["UPDRS3_R"]
    output["UPDRS3_Total"] = table[
        required_groups["bilateral"] + required_groups["right"] + required_groups["left"]
    ].sum(axis=1, min_count=1)
    if "NHY" in table.columns:
        output["NHY"] = table["NHY"]
    if "DYSKIRAT" in table.columns:
        output["Dyskinesia_interference"] = table["DYSKIRAT"]
    if "HRPOSTMED" in table.columns:
        output["Hrs_post_medication"] = table["HRPOSTMED"]
    if "PDSTATE" in table.columns:
        output["UPDRS3_PDSTATE"] = table["PDSTATE"]
    output["Tremor_L"] = table[required_groups["tremor_left"]].sum(axis=1, min_count=1)
    output["Tremor_R"] = table[required_groups["tremor_right"]].sum(axis=1, min_count=1)
    output["Tremor_Asymm"] = output["Tremor_L"] - output["Tremor_R"]
    output["NonTremor_Asymm"] = (
        (output["UPDRS3_L"] - output["Tremor_L"])
        - (output["UPDRS3_R"] - output["Tremor_R"])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path, "done"


def _metadata_study_dates(metadata_csv):
    if metadata_csv is None:
        return None
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        return None
    metadata = pd.read_csv(
        metadata_csv,
        dtype={"RowID": str, "SubjectID": str, "SessionID": str},
        low_memory=False,
        keep_default_na=True,
    )
    required = {"RowID", "StudyDate"}
    if not required <= set(metadata.columns):
        return None
    return metadata[["RowID", "StudyDate"]].copy()


def calculate_disease_duration_metrics(cohort_tables_dir, output_dir, overwrite=False, metadata_csv=None):
    input_path = Path(cohort_tables_dir) / "ppmi_cohort_pd_diagnosis.csv"
    output_path = Path(output_dir) / "ppmi_cohort_calc_disease_duration.csv"
    if output_path.exists() and not overwrite:
        return output_path, "skipped"

    table = _read_cohort_table(input_path)
    if table is None:
        return output_path, "missing"

    required = {"SXDT", "PDDXDT"}
    missing = sorted(required - set(table.columns))
    if missing:
        return output_path, f"missing columns: {', '.join(missing)}"

    study_dates = _metadata_study_dates(metadata_csv)
    if study_dates is None:
        return output_path, "missing metadata StudyDate"

    output = table.loc[:, _base_columns(table)].copy()
    symptoms_onset = pd.to_datetime(table["SXDT"], format="%m/%Y", errors="coerce")
    diagnosis_date = pd.to_datetime(table["PDDXDT"], format="%m/%Y", errors="coerce")
    merged = table[["RowID"]].astype({"RowID": str}).merge(
        study_dates.astype({"RowID": str}),
        how="left",
        on="RowID",
        sort=False,
    )
    imaging_date = pd.to_datetime(merged["StudyDate"], errors="coerce")
    imaging_date = imaging_date.apply(
        lambda value: value.replace(day=1) if pd.notna(value) else pd.NaT
    )

    output["SymptomsOnset"] = symptoms_onset
    output["DiagnosisDate"] = diagnosis_date
    output["SymptomsDuration_days"] = (imaging_date - symptoms_onset).dt.days
    output["DiagnosisDuration_days"] = (imaging_date - diagnosis_date).dt.days

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path, "done"


def calculate_rbd_score_metrics(cohort_tables_dir, output_dir, overwrite=False):
    input_path = Path(cohort_tables_dir) / "ppmi_cohort_rbd.csv"
    output_path = Path(output_dir) / "ppmi_cohort_calc_rbd_score.csv"
    if output_path.exists() and not overwrite:
        return output_path, "skipped"

    table = _read_cohort_table(input_path)
    if table is None:
        return output_path, "missing"

    item_1_12 = [
        "DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR", "DRMVERBL",
        "DRMFIGHT", "DRMUMV", "DRMOBJFL", "MVAWAKEN", "DRMREMEM", "SLPDSTRB",
    ]
    item_13 = ["STROKE", "HETRA", "PARKISM", "RLS", "NARCLPSY", "DEPRS", "EPILEPSY", "BRNINFM", "CNSOTH"]
    missing = sorted((set(item_1_12) | set(item_13)) - set(table.columns))
    if missing:
        return output_path, f"missing columns: {', '.join(missing)}"

    output = table.loc[:, _base_columns(table)].copy()
    score_1 = table[item_1_12].sum(axis=1, min_count=1)
    score_2 = table[item_13].sum(axis=1, min_count=1)
    score_2[score_2 > 0] = 1
    output["RBD_score"] = score_1 + score_2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path, "done"


def calculate_cohort_metrics(cohort_tables_dir, output_dir=None, overwrite=False, metadata_csv=None):
    cohort_tables_dir = Path(cohort_tables_dir)
    output_dir = Path(output_dir) if output_dir is not None else cohort_tables_dir / "calculated"
    output_dir.mkdir(parents=True, exist_ok=True)

    calculators = {
        "motor": calculate_motor_metrics,
        "disease_duration": calculate_disease_duration_metrics,
        "rbd_score": calculate_rbd_score_metrics,
    }
    results = {}
    for name, calculator in calculators.items():
        kwargs = {"metadata_csv": metadata_csv} if name == "disease_duration" else {}
        results[name] = calculator(cohort_tables_dir, output_dir, overwrite=overwrite, **kwargs)
    return results


def build_cohort_clinical_tables(
    metadata_csv,
    study_tables_root,
    output_dir,
    table_names=None,
    overwrite=False,
    calculate_metrics=True,
):
    outputs, skipped = write_cohort_tables(
        metadata_csv=metadata_csv,
        study_tables_root=study_tables_root,
        output_dir=output_dir,
        table_names=table_names,
        overwrite=overwrite,
    )
    qa_path, qa_status = write_imaging_qa_table(
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    metric_results = {}
    if calculate_metrics:
        metric_results = calculate_cohort_metrics(
            output_dir,
            overwrite=overwrite,
            metadata_csv=metadata_csv,
        )
    return {
        "tables": outputs,
        "skipped": skipped,
        "imaging_qa": (qa_path, qa_status),
        "metrics": metric_results,
    }
