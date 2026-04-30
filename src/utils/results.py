import csv
import os


def create_exp_directory(save_dir):
    exp_model_dir = os.path.join(save_dir, "models")
    if not os.path.exists(exp_model_dir):
        os.makedirs(exp_model_dir)

    results_csv = "results.csv"
    with open(os.path.join(save_dir, results_csv), "w") as f:
        f.write(
            "epoch,train_loss,val_loss,dice,Jac,clDice,precision,accuracy,betti_error_0,betti_error_1,elapsed_hms\n"
        )


def write_epoch_result_row(save_dir, epoch, metrics, elapsed_hms=""):
    results_csv = "results.csv"
    with open(os.path.join(save_dir, results_csv), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                f"{int(epoch):03d}",
                f'{float(metrics["train_loss"]):0.6f}',
                f'{float(metrics["val_loss"]):0.6f}',
                f'{float(metrics["dice"]):0.6f}',
                f'{float(metrics["jac"]):0.6f}',
                f'{float(metrics["cldice"]):0.6f}',
                f'{float(metrics.get("precision", float("nan"))):0.6f}',
                f'{float(metrics.get("accuracy", float("nan"))):0.6f}',
                f'{float(metrics["betti_error_0"]):0.6f}',
                f'{float(metrics["betti_error_1"]):0.6f}',
                elapsed_hms or "",
            ]
        )


def write_eval_summary(
    save_dir,
    split_name,
    metrics,
    checkpoint_name="",
    evaluated_split="",
    eval_epoch="",
    elapsed_hms="",
):
    summary_csv = os.path.join(save_dir, f"{split_name}_results.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "result_type",
                "evaluated_split",
                "eval_epoch",
                "checkpoint",
                "train_loss",
                "eval_loss",
                "dice",
                "jac",
                "cldice",
                "precision",
                "accuracy",
                "betti_error_0",
                "betti_error_1",
                "elapsed_hms",
            ]
        )
        writer.writerow(
            [
                split_name,
                evaluated_split,
                eval_epoch,
                checkpoint_name,
                f'{float(metrics["train_loss"]):0.6f}',
                f'{float(metrics["val_loss"]):0.6f}',
                f'{float(metrics["dice"]):0.6f}',
                f'{float(metrics["jac"]):0.6f}',
                f'{float(metrics["cldice"]):0.6f}',
                f'{float(metrics.get("precision", float("nan"))):0.6f}',
                f'{float(metrics.get("accuracy", float("nan"))):0.6f}',
                f'{float(metrics["betti_error_0"]):0.6f}',
                f'{float(metrics["betti_error_1"]):0.6f}',
                elapsed_hms or "",
            ]
        )


def save_best_checkpoint(
    save_dir,
    model_state_dict,
    optimizer_state_dict,
    current_epoch,
    stop_state,
    monitor_metric,
    epoch_metrics,
):
    best_model_dir = os.path.join(save_dir, "models")
    best_model_path = os.path.join(best_model_dir, "best_model.pth")
    checkpoint_best_path = os.path.join(best_model_dir, "checkpoint_best.pth")

    import torch

    torch.save(model_state_dict, best_model_path)
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "epoch": int(current_epoch),
            "best_val_dice": float(stop_state["best_val_dice"]),
            "best_val_loss": float(stop_state["best_val_loss"]),
            "monitor_metric": monitor_metric,
            "monitor_value": float(stop_state["best_monitor"]),
            "tie_break_with_loss": bool(stop_state["used_tie_break"]),
        },
        checkpoint_best_path,
    )
    with open(os.path.join(best_model_dir, "best_model_meta.txt"), "w") as f:
        f.write(f"best_epoch={current_epoch}\n")
        f.write(f'best_val_dice={stop_state["best_val_dice"]:.6f}\n')
        f.write(f'best_val_loss={stop_state["best_val_loss"]:.6f}\n')
        f.write(f"monitor_metric={monitor_metric}\n")
        f.write(f'best_monitor_value={stop_state["best_monitor"]:.6f}\n')
        f.write(f'used_tie_break_with_loss={bool(stop_state["used_tie_break"])}\n')
        f.write(f'best_train_loss={epoch_metrics["train_loss"]:.6f}\n')
        f.write(f'best_jac={epoch_metrics["jac"]:.6f}\n')
        f.write(f'best_clDice={epoch_metrics["cldice"]:.6f}\n')
        f.write(f'best_betti_error_0={epoch_metrics["betti_error_0"]:.6f}\n')
        f.write(f'best_betti_error_1={epoch_metrics["betti_error_1"]:.6f}\n')
