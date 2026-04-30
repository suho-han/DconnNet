from connect_loss import connect_loss


def build_loss_functions(
    args,
    hori_translation,
    verti_translation,
    label_mode,
    fusion_enabled,
):
    if fusion_enabled:
        loss_func = connect_loss(
            args,
            hori_translation,
            verti_translation,
            label_mode=label_mode,
            conn_num=8,
            sigma=args.sigma,
            conn_layout="standard8",
        )
        loss_func_outer = connect_loss(
            args,
            hori_translation,
            verti_translation,
            label_mode=label_mode,
            conn_num=8,
            sigma=args.sigma,
            conn_layout="out8",
        )
        return loss_func, loss_func_outer

    loss_func = connect_loss(
        args,
        hori_translation,
        verti_translation,
        label_mode=label_mode,
        conn_num=args.conn_num,
        sigma=args.sigma,
        conn_layout=getattr(args, "conn_layout", None),
    )
    return loss_func, None
