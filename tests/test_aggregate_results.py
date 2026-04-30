import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_results import (
    aggregate_output_stem,
    _experiment_mean_unique_key,
    discover_target_roots,
    experiment_sort_key,
    filter_rows_for_aggregate_scope,
    parse_experiment_metadata,
    should_include_dataset_in_default_aggregate,
    write_ablation_latex,
    write_experiment_mean_latex,
    write_experiment_mean_dataset_tables_latex,
)


def _row(
    dataset,
    experiment,
    conn_layout="default",
    conn_fusion="none",
    fusion_loss_profile="A",
    seg_aux_weight=None,
    seg_aux_variant="none",
):
    return {
        "dataset": dataset,
        "experiment": experiment,
        "conn_num": 8,
        "conn_layout": conn_layout,
        "conn_fusion": conn_fusion,
        "fusion_loss_profile": fusion_loss_profile,
        "seg_aux_weight": seg_aux_weight,
        "seg_aux_variant": seg_aux_variant,
        "loss": "bce",
        "num_folds": 5,
        "best_dice": 0.9,
        "best_dice_std": 0.01,
        "best_jac": 0.8,
        "best_jac_std": 0.01,
        "best_accuracy": 0.95,
        "best_accuracy_std": 0.01,
        "best_precision": 0.96,
        "best_precision_std": 0.01,
        "best_cldice": 0.7,
        "best_cldice_std": 0.01,
        "best_betti_error_0": 0.1,
        "best_betti_error_0_std": 0.01,
        "best_betti_error_1": 0.2,
        "best_betti_error_1_std": 0.01,
    }


def test_experiment_mean_dataset_tables_are_grouped_by_dataset_family(tmp_path):
    rows_by_dataset = {
        "isic": [_row("isic", "isic_exp")],
        "chase": [_row("chase", "chase_exp")],
        "drive": [_row("drive", "drive_exp")],
        "cremi": [_row("cremi", "cremi_exp")],
        "octa500-6M": [_row("octa500-6M", "octa6m_exp")],
        "octa500-3M": [_row("octa500-3M", "octa3m_exp")],
    }

    out_path = Path(tmp_path) / "grouped.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()

    cremi_idx = text.index(r"\section*{CREMI}")
    drive_idx = text.index(r"\section*{DRIVE}")
    chase_idx = text.index(r"\section*{CHASE}")
    isic_idx = text.index(r"\section*{ISIC}")
    octa_idx = text.index(r"\section*{OCTA3M\&6M}")

    assert cremi_idx < drive_idx < chase_idx < isic_idx < octa_idx
    assert text.count(r"\clearpage") >= 5


def test_default_aggregate_scope_includes_only_drive_chase_and_octa_family():
    assert should_include_dataset_in_default_aggregate("drive")
    assert should_include_dataset_in_default_aggregate("CHASEDB1")
    assert should_include_dataset_in_default_aggregate("octa500-3M")
    assert should_include_dataset_in_default_aggregate("octa500-6M")
    assert not should_include_dataset_in_default_aggregate("cremi")
    assert not should_include_dataset_in_default_aggregate("isic2018")


def test_filter_rows_for_aggregate_scope_defaults_to_drive_chase_octa():
    rows = [
        _row("drive", "drive_exp"),
        _row("chase", "chase_exp"),
        _row("octa500-3M", "octa_exp"),
        _row("cremi", "cremi_exp"),
        _row("isic", "isic_exp"),
    ]

    filtered = filter_rows_for_aggregate_scope(rows, include_all=False)

    assert [row["dataset"] for row in filtered] == ["drive", "chase", "octa500-3M"]


def test_filter_rows_for_aggregate_scope_with_all_keeps_all_counted_datasets():
    rows = [
        _row("drive", "drive_exp"),
        _row("chase", "chase_exp"),
        _row("octa500-3M", "octa_exp"),
        _row("cremi", "cremi_exp"),
        _row("isic", "isic_exp"),
        _row("trash", "trash_exp"),
    ]

    filtered = filter_rows_for_aggregate_scope(rows, include_all=True)

    assert [row["dataset"] for row in filtered] == [
        "drive",
        "chase",
        "octa500-3M",
        "cremi",
        "isic",
    ]


def test_aggregate_output_stem_appends_all_suffix_only_for_all_mode():
    assert aggregate_output_stem("summary", include_all=False) == "summary_experiment_means"
    assert aggregate_output_stem("summary", include_all=True) == "summary_experiment_means_all"
    assert (
        aggregate_output_stem("summary_experiment_means", include_all=True)
        == "summary_experiment_means_all"
    )


def test_discover_target_roots_excludes_smoke_directory(tmp_path):
    real_root = Path(tmp_path) / "drive" / "dist_8_smooth_l1"
    smoke_root = Path(tmp_path) / "_smoke" / "drive" / "dist_8_smooth_l1"
    real_root.mkdir(parents=True, exist_ok=True)
    smoke_root.mkdir(parents=True, exist_ok=True)
    (real_root / "final_results_1.csv").write_text("best_epoch,best_dice\n1,0.9\n")
    (smoke_root / "final_results_1.csv").write_text("best_epoch,best_dice\n1,0.1\n")

    roots = discover_target_roots(str(tmp_path), ["1"], "final_results_{fold}.csv")
    roots_set = {Path(p).resolve() for p in roots}

    assert real_root.resolve() in roots_set
    assert smoke_root.resolve() not in roots_set


def test_out8_conn_label_is_rendered_as_prime(tmp_path):
    rows_by_dataset = {
        "drive": [_row("drive", "binary", conn_layout="out8")],
    }

    out_path = Path(tmp_path) / "out8.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"binary & 8' & none & none & bce" in text


def test_fusion_experiment_metadata_is_parsed_and_sorted_by_mode_prefix():
    meta = parse_experiment_metadata("dist_scaled_sum_C_rs0.3_8_smooth_l1")
    assert meta["experiment"] == "dist_scaled_sum_C_rs0.3"
    assert meta["label_mode"] == "dist"
    assert meta["conn_num"] == 8
    assert meta["conn_layout"] == "default"
    assert meta["loss"] == "smooth_l1"
    assert meta["conn_fusion"] == "scaled_sum"
    assert meta["fusion_loss_profile"] == "C"

    key = experiment_sort_key(
        {
            "experiment": meta["experiment"],
            "conn_num": meta["conn_num"],
            "conn_layout": meta["conn_layout"],
            "loss": meta["loss"],
            "fold_scope": "direct",
            "fold_scope_count": "NA",
        }
    )
    assert key[0] == 1  # dist family ordering

def test_decoder_guided_is_parsed_as_fusion_objective():
    meta = parse_experiment_metadata("dist_inverted_decoder_guided_A_8_gjml_sf_l1")
    assert meta["experiment"] == "dist_inverted_decoder_guided_A"
    assert meta["label_mode"] == "dist_inverted"
    assert meta["conn_num"] == 8
    assert meta["conn_layout"] == "default"
    assert meta["loss"] == "gjml_sf_l1"
    assert meta["conn_fusion"] == "dg"
    assert meta["fusion_loss_profile"] == "A"


def test_ordered_name_with_segaux_suffix_keeps_conn_and_loss_metadata():
    meta = parse_experiment_metadata("dist_inverted_decoder_guided_A_8_gjml_sf_l1_segaux")
    assert meta["experiment"] == "dist_inverted_decoder_guided_A"
    assert meta["label_mode"] == "dist_inverted"
    assert meta["conn_num"] == 8
    assert meta["conn_layout"] == "default"
    assert meta["loss"] == "gjml_sf_l1"
    assert meta["conn_fusion"] == "dg"
    assert meta["fusion_loss_profile"] == "A"
    assert meta["seg_aux_weight"] is None
    assert meta["seg_aux_variant"] == "segaux"


def test_ordered_name_with_segaux_weight_suffix_parses_numeric_weight():
    meta = parse_experiment_metadata("dist_inverted_decoder_guided_A_8_gjml_sf_l1_segaux_w0.1")
    assert meta["seg_aux_weight"] == 0.1
    assert meta["seg_aux_variant"] == "w0.1"


def test_dg_direct_is_parsed_as_distinct_fusion_objective():
    meta = parse_experiment_metadata("binary_dg_direct_A_8_bce_segaux_w0.5")
    assert meta["experiment"] == "binary_dg_direct_A"
    assert meta["label_mode"] == "binary"
    assert meta["conn_num"] == 8
    assert meta["loss"] == "bce"
    assert meta["conn_fusion"] == "dg_direct"
    assert meta["fusion_loss_profile"] == "A"
    assert meta["seg_aux_weight"] == 0.5
    assert meta["seg_aux_variant"] == "w0.5"


def test_dataset_table_keeps_dg_in_fusion_column(tmp_path):
    rows_by_dataset = {
        "cremi": [
            _row(
                "cremi",
                "dist_inverted_decoder_guided_A",
                conn_fusion="dg",
                fusion_loss_profile="A",
            )
        ],
    }

    out_path = Path(tmp_path) / "dgrf.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"dist\_inverted & 8 & dg & none & bce" in text


def test_dataset_table_shows_segaux_weight_in_dec_column(tmp_path):
    rows_by_dataset = {
        "drive": [
            _row(
                "drive",
                "dist_inverted_decoder_guided_A",
                conn_fusion="dg",
                fusion_loss_profile="A",
                seg_aux_weight=0.1,
                seg_aux_variant="w0.1",
            )
        ],
    }

    out_path = Path(tmp_path) / "segaux_dec.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"dist\_inverted & 8 & dg & w0.1 & bce" in text


def test_dataset_table_shows_default_segaux_in_dec_column(tmp_path):
    rows_by_dataset = {
        "drive": [
            _row(
                "drive",
                "dist_inverted_decoder_guided_A",
                conn_fusion="dg",
                fusion_loss_profile="A",
                seg_aux_weight=None,
                seg_aux_variant="segaux",
            )
        ],
    }

    out_path = Path(tmp_path) / "segaux_default_dec.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"dist\_inverted & 8 & dg & segaux & bce" in text


def test_ablation_table_hides_dg_profile_a(tmp_path):
    rows = [
        _row(
            "drive",
            "dist_inverted_decoder_guided_A",
            conn_fusion="dg",
            fusion_loss_profile="A",
        )
    ]

    out_path = Path(tmp_path) / "ablation_decoder_guided.tex"
    write_ablation_latex(str(out_path), rows)

    text = out_path.read_text()
    assert r"\multirow{2}{*}{No.}" in text
    assert r"dg &  & none" in text
    assert r"dg & A & none" not in text


def test_dataset_table_hides_scaled_sum_profile_a_but_keeps_rs(tmp_path):
    rows_by_dataset = {
        "drive": [
            _row(
                "drive",
                "dist_scaled_sum_A_rs0.1",
                conn_fusion="scaled_sum",
                fusion_loss_profile="A",
            )
        ],
    }
    rows_by_dataset["drive"][0]["fusion_residual_scale"] = 0.1

    out_path = Path(tmp_path) / "scaled_sum_summary.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"dist & 8 & scaled\_sum/rs0.1 & none & bce" in text
    assert r"scaled\_sum/A/rs0.1" not in text


def test_ablation_table_hides_scaled_sum_profile_a_but_keeps_rs(tmp_path):
    rows = [
        _row(
            "drive",
            "dist_scaled_sum_A_rs0.1",
            conn_fusion="scaled_sum",
            fusion_loss_profile="A",
        )
    ]
    rows[0]["fusion_residual_scale"] = 0.1

    out_path = Path(tmp_path) / "ablation_scaled_sum.tex"
    write_ablation_latex(str(out_path), rows)

    text = out_path.read_text()
    assert r"scaled\_sum & rs0.1 & none" in text
    assert r"scaled\_sum & A/rs0.1 & none" not in text


def test_experiment_sort_key_follows_label_conn_fusion_dec_loss_order():
    row_a = _row(
        "drive",
        "dist_exp",
        conn_fusion="decoder_guided",
        fusion_loss_profile="A",
    )
    row_a["loss"] = "gjml_sf_l1"

    row_b = _row(
        "drive",
        "dist_exp",
        conn_fusion="none",
        fusion_loss_profile="A",
    )
    row_b["loss"] = "smooth_l1"

    # With requested order (label_mode, conn, fusion, SegAux, loss),
    # "fusion=none" should come before decoder-guided entry.
    assert experiment_sort_key(row_b) < experiment_sort_key(row_a)


def test_experiment_mean_unique_key_keeps_suffix_variants_distinct():
    base = _row(
        "cremi",
        "dist_inverted_decoder_guided_A",
        conn_fusion="decoder_guided",
        fusion_loss_profile="A",
    )
    base["loss"] = "gjml_sf_l1"
    base["fold_scope"] = "direct"
    base["experiment_source"] = "dist_inverted_decoder_guided_A_8_gjml_sf_l1"

    segaux = dict(base)
    segaux["experiment_source"] = "dist_inverted_decoder_guided_A_8_gjml_sf_l1_segaux"

    assert _experiment_mean_unique_key(base) != _experiment_mean_unique_key(segaux)


def test_ablation_latex_includes_conn_fusion_and_fusion_loss_table(tmp_path):
    rows = [
        _row(
            "drive",
            "binary_gate_A",
            conn_fusion="gate",
            fusion_loss_profile="A",
        ),
        _row(
            "drive",
            "binary_gate_C",
            conn_fusion="gate",
            fusion_loss_profile="C",
        ),
        _row(
            "drive",
            "binary_conv_residual_C",
            conn_fusion="conv_residual",
            fusion_loss_profile="C",
        ),
    ]

    out_path = Path(tmp_path) / "ablation.tex"
    write_ablation_latex(str(out_path), rows)

    text = out_path.read_text()
    assert r"\caption{Ablation on Fusion + Decoder/SegAux (CREMI, DRIVE)}" in text
    assert r"\multirow{2}{*}{Conn. Fusion}" in text
    assert r"\multirow{2}{*}{Fusion Spec}" in text
    assert r"gate & A" in text
    assert r"gate & C" in text
    assert r"conv\_residual & C" in text


def test_ablation_label_mode_table_keeps_only_three_modes(tmp_path):
    rows = [
        _row("drive", "binary_gate_A", conn_fusion="gate", fusion_loss_profile="A"),
        _row("drive", "dist_gate_A", conn_fusion="gate", fusion_loss_profile="A"),
        _row(
            "drive",
            "dist_inverted_gate_C",
            conn_fusion="gate",
            fusion_loss_profile="C",
        ),
    ]

    out_path = Path(tmp_path) / "ablation_modes.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    assert r"\caption{Ablation on Label Mode (CREMI, DRIVE)}" in text
    assert r"binary &" in text
    assert r"dist &" in text
    assert r"dist\_inverted &" in text
    assert r"binary\_gate\_A" not in text
    assert r"dist\_inverted\_gate\_C\_dec\_residual\_gate" not in text
    assert r"\caption{Ablation on Fusion}" not in text
    assert r"\caption{Ablation on Fusion + Decoder/SegAux (CREMI, DRIVE)}" in text


def test_ablation_fusion_objective_table_order_is_fixed(tmp_path):
    rows = [
        _row("drive", "binary_none", conn_fusion="none", fusion_loss_profile="A"),
        _row(
            "drive",
            "binary_scaled_sum_A",
            conn_fusion="scaled_sum",
            fusion_loss_profile="A",
        ),
        _row(
            "drive",
            "binary_gate_A",
            conn_fusion="gate",
            fusion_loss_profile="A",
        ),
        _row(
            "drive",
            "binary_conv_residual_A",
            conn_fusion="conv_residual",
            fusion_loss_profile="A",
        ),
    ]

    out_path = Path(tmp_path) / "ablation_fusion_order.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    assert r"\caption{Ablation on Fusion}" not in text
    assert r"\caption{Ablation on Fusion + Decoder/SegAux (CREMI, DRIVE)}" in text
    none_idx = text.index(r"none & A")
    conv_idx = text.index(r"conv\_residual & A")
    gate_idx = text.index(r"gate & A")
    scaled_idx = text.index(r"scaled\_sum &  & none")
    assert none_idx < conv_idx < gate_idx < scaled_idx


def test_ablation_latex_splits_tables_into_cremi_drive_and_other(tmp_path):
    rows = [
        _row("cremi", "dist", conn_fusion="none", fusion_loss_profile="A"),
        _row("drive", "dist", conn_fusion="none", fusion_loss_profile="A"),
        _row("isic", "dist", conn_fusion="none", fusion_loss_profile="A"),
    ]
    for row in rows:
        row["loss"] = "smooth_l1"

    out_path = Path(tmp_path) / "ablation_split.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    assert r"\caption{Ablation on Loss (CREMI, DRIVE)}" in text
    assert r"\caption{Ablation on Loss (Other datasets)}" in text
    assert r"\caption{Ablation on Label Mode (CREMI, DRIVE)}" in text
    assert r"\caption{Ablation on Label Mode (Other datasets)}" in text

    other_caption_idx = text.index(r"\caption{Ablation on Loss (Other datasets)}")
    other_table_start = text.rfind(r"\begin{tabular}", 0, other_caption_idx)
    other_table_text = text[other_table_start:other_caption_idx]
    assert r"\multicolumn{2}{c}{isic}" in other_table_text
    assert r"\multicolumn{2}{c}{cremi}" not in other_table_text
    assert r"\multicolumn{2}{c}{drive}" not in other_table_text


def test_ablation_drop_one_best_and_loss_toggle_use_only_smooth_or_gjml(tmp_path):
    rows = []

    bce_row = _row("drive", "binary_base")
    bce_row["loss"] = "bce"
    bce_row["best_dice"] = 0.99
    bce_row["best_jac"] = 0.89
    rows.append(bce_row)

    smooth_row = _row("drive", "dist_smooth")
    smooth_row["loss"] = "smooth_l1"
    smooth_row["best_dice"] = 0.95
    smooth_row["best_jac"] = 0.85
    rows.append(smooth_row)

    gjml_row = _row("drive", "dist_gjml")
    gjml_row["loss"] = "gjml_sf_l1"
    gjml_row["best_dice"] = 0.93
    gjml_row["best_jac"] = 0.83
    rows.append(gjml_row)

    out_path = Path(tmp_path) / "ablation_drop_one_loss.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    caption = r"\caption{Best-run drop-one deltas (DRIVE)}"
    caption_idx = text.index(caption)
    table_start = text.rfind(r"\begin{tabular}", 0, caption_idx)
    table_text = text[table_start:caption_idx]

    assert r"No. & Dataset & Variant" in table_text
    assert r"1 & drive & BEST" in table_text
    assert r"drive & BEST & dist & 8 & none & none & smooth\_l1" in table_text
    assert r"drive & -Loss & dist & 8 & none & none & gjml\_sf\_l1" in table_text
    assert r"drive & BEST & binary & 8 & none & none & bce" not in table_text


def test_ablation_drop_one_fusion_uses_same_conn_plain_none_baseline(tmp_path):
    best_row = _row(
        "drive",
        "dist_inverted_best",
        conn_fusion="decoder_guided",
        fusion_loss_profile="A",
        seg_aux_weight=0.5,
        seg_aux_variant="w0.5",
    )
    best_row["loss"] = "smooth_l1"
    best_row["best_dice"] = 0.95
    best_row["best_jac"] = 0.85

    same_conn_plain = _row("drive", "dist_inverted_plain")
    same_conn_plain["loss"] = "smooth_l1"
    same_conn_plain["best_dice"] = 0.93
    same_conn_plain["best_jac"] = 0.83

    better_wrong_conn = _row("drive", "dist_inverted_plain_24")
    better_wrong_conn["conn_num"] = 24
    better_wrong_conn["loss"] = "smooth_l1"
    better_wrong_conn["best_dice"] = 0.94
    better_wrong_conn["best_jac"] = 0.84

    out_path = Path(tmp_path) / "ablation_drop_one_fusion.tex"
    write_ablation_latex(str(out_path), [best_row, same_conn_plain, better_wrong_conn])
    text = out_path.read_text()

    caption = r"\caption{Best-run drop-one deltas (DRIVE)}"
    caption_idx = text.index(caption)
    table_start = text.rfind(r"\begin{tabular}", 0, caption_idx)
    table_text = text[table_start:caption_idx]

    assert r"drive & -Fusion & dist\_inverted & 8 & none & none & smooth\_l1" in table_text
    assert r"drive & -Fusion & dist\_inverted & 24 & none & none & smooth\_l1" not in table_text


def test_ablation_drop_one_fusion_is_blank_when_best_is_already_none(tmp_path):
    best_row = _row("isic", "dist_inverted_best")
    best_row["loss"] = "smooth_l1"
    best_row["best_dice"] = 0.95
    best_row["best_jac"] = 0.85

    alt_row = _row(
        "isic",
        "dist_inverted_gate",
        conn_fusion="gate",
        fusion_loss_profile="A",
    )
    alt_row["loss"] = "smooth_l1"
    alt_row["best_dice"] = 0.94
    alt_row["best_jac"] = 0.84

    out_path = Path(tmp_path) / "ablation_drop_one_fusion_none.tex"
    write_ablation_latex(str(out_path), [best_row, alt_row])
    text = out_path.read_text()

    caption = r"\caption{Best-run drop-one deltas (isic)}"
    caption_idx = text.index(caption)
    table_start = text.rfind(r"\begin{tabular}", 0, caption_idx)
    table_text = text[table_start:caption_idx]

    assert r"isic & -Fusion & - & - & - & - & -" in table_text


def test_ablation_drop_one_splits_other_datasets_per_dataset(tmp_path):
    isic_row = _row("isic", "dist_isic")
    isic_row["loss"] = "smooth_l1"
    chase_row = _row("chase", "dist_chase")
    chase_row["loss"] = "smooth_l1"

    out_path = Path(tmp_path) / "ablation_drop_one_other_split.tex"
    write_ablation_latex(str(out_path), [isic_row, chase_row])
    text = out_path.read_text()

    assert r"\caption{Best-run drop-one deltas (isic)}" in text
    assert r"\caption{Best-run drop-one deltas (chase)}" in text
    assert r"\caption{Best-run drop-one deltas (Other datasets)}" not in text


def test_ablation_drop_one_label_mode_is_blank_when_best_is_already_binary(tmp_path):
    best_row = _row("cremi", "binary_best")
    best_row["loss"] = "smooth_l1"
    best_row["best_dice"] = 0.95
    best_row["best_jac"] = 0.85

    alt_row = _row("cremi", "dist_alt")
    alt_row["loss"] = "smooth_l1"
    alt_row["best_dice"] = 0.94
    alt_row["best_jac"] = 0.84

    out_path = Path(tmp_path) / "ablation_drop_one_binary_label.tex"
    write_ablation_latex(str(out_path), [best_row, alt_row])
    text = out_path.read_text()

    caption = r"\caption{Best-run drop-one deltas (CREMI)}"
    caption_idx = text.index(caption)
    table_start = text.rfind(r"\begin{tabular}", 0, caption_idx)
    table_text = text[table_start:caption_idx]

    assert r"cremi & -Label Mode & - & - & - & - & -" in table_text


def test_ablation_drop_one_appends_fixed_binary8_plain_base_row(tmp_path):
    best_row = _row(
        "drive",
        "dist_inverted_best",
        conn_fusion="decoder_guided",
        fusion_loss_profile="A",
        seg_aux_weight=0.5,
        seg_aux_variant="w0.5",
    )
    best_row["label_mode"] = "dist_inverted"
    best_row["loss"] = "smooth_l1"
    best_row["best_dice"] = 0.95
    best_row["best_jac"] = 0.85

    baseline_row = _row("drive", "binary_8_base")
    baseline_row["label_mode"] = "binary"
    baseline_row["loss"] = "bce"
    baseline_row["best_dice"] = 0.90
    baseline_row["best_jac"] = 0.80

    out_path = Path(tmp_path) / "ablation_drop_one_base.tex"
    write_ablation_latex(str(out_path), [best_row, baseline_row])
    text = out_path.read_text()

    caption = r"\caption{Best-run drop-one deltas (DRIVE)}"
    caption_idx = text.index(caption)
    table_start = text.rfind(r"\begin{tabular}", 0, caption_idx)
    table_text = text[table_start:caption_idx]

    assert r"drive & BASE & binary & 8 & none & none & bce" in table_text
    assert table_text.index(r"drive & -SegAux") < table_text.index(r"drive & -Fusion")
    assert table_text.index(r"drive & -SegAux") < table_text.index(r"drive & BASE & binary & 8 & none & none & bce")


def test_ablation_drop_one_includes_cldice_and_betti_with_error_direction(tmp_path):
    best_row = _row("drive", "dist_best")
    best_row["loss"] = "smooth_l1"
    best_row["best_dice"] = 0.95
    best_row["best_jac"] = 0.85
    best_row["best_cldice"] = 0.80
    best_row["best_betti_error_0"] = 0.30
    best_row["best_betti_error_1"] = 0.40

    alt_row = _row("drive", "dist_gjml")
    alt_row["loss"] = "gjml_sf_l1"
    alt_row["best_dice"] = 0.93
    alt_row["best_jac"] = 0.83
    alt_row["best_cldice"] = 0.78
    alt_row["best_betti_error_0"] = 0.20
    alt_row["best_betti_error_1"] = 0.55

    out_path = Path(tmp_path) / "ablation_drop_one_metrics.tex"
    write_ablation_latex(str(out_path), [best_row, alt_row])
    text = out_path.read_text()

    assert r"clDice & Err $(\beta_0)$ & Err $(\beta_1)$" in text
    assert r"\textcolor{green!60!black}{-0.1000}" in text
    assert r"\textcolor{red!70!black}{+0.1500}" in text


def test_dataset_tables_ignore_trash_dataset(tmp_path):
    rows_by_dataset = {
        "drive": [_row("drive", "drive_exp")],
        "trash": [_row("trash", "trash_exp")],
    }

    out_path = Path(tmp_path) / "datasets_no_trash.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert "trash" not in text
    assert r"\caption{Cross-experiment mean summary (drive, Loss=bce)}" in text


def test_default_scope_dataset_summary_excludes_cremi_and_isic_sections(tmp_path):
    rows = [
        _row("drive", "drive_exp"),
        _row("chase", "chase_exp"),
        _row("octa500-3M", "octa_exp"),
        _row("cremi", "cremi_exp"),
        _row("isic", "isic_exp"),
    ]
    filtered = filter_rows_for_aggregate_scope(rows, include_all=False)
    rows_by_dataset = {}
    for row in filtered:
        rows_by_dataset.setdefault(row["dataset"], []).append(row)

    out_path = Path(tmp_path) / "default_scope_datasets.tex"
    write_experiment_mean_dataset_tables_latex(
        str(out_path),
        "Grouped summary",
        rows_by_dataset,
    )

    text = out_path.read_text()
    assert r"\section*{DRIVE}" in text
    assert r"\section*{CHASE}" in text
    assert r"\section*{OCTA3M\&6M}" in text
    assert r"\section*{CREMI}" not in text
    assert r"\section*{ISIC}" not in text
    assert text.count(r"\clearpage") >= 2


def test_ablation_tables_ignore_trash_dataset(tmp_path):
    rows = [
        _row("drive", "dist_drive"),
        _row("trash", "dist_trash"),
    ]
    for row in rows:
        row["loss"] = "smooth_l1"

    out_path = Path(tmp_path) / "ablation_no_trash.tex"
    write_ablation_latex(str(out_path), rows)
    text = out_path.read_text()

    assert "trash" not in text
    assert r"\caption{Ablation on Loss (CREMI, DRIVE)}" in text
    assert r"\caption{Best-run drop-one deltas (DRIVE)}" in text


def test_default_scope_ablation_excludes_cremi_and_isic(tmp_path):
    rows = [
        _row("drive", "dist_drive"),
        _row("chase", "dist_chase"),
        _row("octa500-3M", "dist_octa"),
        _row("cremi", "dist_cremi"),
        _row("isic", "dist_isic"),
    ]
    for row in rows:
        row["loss"] = "smooth_l1"

    filtered = filter_rows_for_aggregate_scope(rows, include_all=False)
    out_path = Path(tmp_path) / "default_scope_ablation.tex"
    write_ablation_latex(str(out_path), filtered)
    text = out_path.read_text()

    assert r"\multicolumn{2}{c}{drive}" in text
    assert r"\multicolumn{2}{c}{chase}" in text
    assert r"\multicolumn{2}{c}{octa500--3M}" in text
    assert r"\multicolumn{2}{c}{cremi}" not in text
    assert r"\multicolumn{2}{c}{isic}" not in text
    assert r"\clearpage" not in text


def test_cross_experiment_latex_ignores_trash_dataset(tmp_path):
    rows = [
        _row("drive", "drive_exp"),
        _row("trash", "trash_exp"),
    ]

    out_path = Path(tmp_path) / "all_no_trash.tex"
    write_experiment_mean_latex(
        str(out_path),
        "Cross-experiment Mean Summary",
        rows,
    )

    text = out_path.read_text()
    assert "trash" not in text
    assert r"\caption{Cross-experiment mean summary (drive)}" in text
