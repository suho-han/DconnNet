def compose_fusion_profile_loss_terms(
    profile,
    lambda_inner,
    lambda_outer,
    lambda_fused,
    fused_terms,
    inner_terms,
    outer_terms,
):
    profile_name = str(profile).upper()
    if profile_name not in {"A", "B", "C"}:
        raise ValueError(f"Unsupported fusion_loss_profile={profile}")

    seg = fused_terms["vote"] + fused_terms["dice"]
    fused_affinity = fused_terms["affinity"]

    total = seg + float(lambda_fused) * fused_affinity
    terms = {
        "total": total,
        "seg": seg,
        "vote": fused_terms["vote"],
        "dice": fused_terms["dice"],
        "fused_affinity": fused_affinity,
    }

    if profile_name in {"B", "C"}:
        total = total + float(lambda_inner) * inner_terms["affinity"]
        terms["inner_affinity"] = inner_terms["affinity"]
    if profile_name == "C":
        total = total + float(lambda_outer) * outer_terms["affinity"]
        terms["outer_affinity"] = outer_terms["affinity"]

    terms["total"] = total
    return total, terms
