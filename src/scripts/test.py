import os


def run_test_only_eval(solver, net, val_loader, test_loader, writer):
    eval_split_name = "test" if test_loader is not None else "val"
    print(f"START {eval_split_name.upper()}-ONLY EVAL.")
    test_metrics = solver.test_epoch(
        net,
        val_loader if test_loader is None else test_loader,
        0,
        split_name=eval_split_name,
    )
    solver._write_epoch_result_row(1, test_metrics, elapsed_hms="")
    solver._write_eval_summary(
        "final",
        test_metrics,
        checkpoint_name=os.path.basename(solver.args.pretrained) if solver.args.pretrained else "",
        evaluated_split=eval_split_name,
        eval_epoch=1,
        elapsed_hms="",
    )
    writer.close()
