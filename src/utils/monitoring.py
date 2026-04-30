import math


def is_nan_metric(value):
    return math.isnan(float(value))


def format_elapsed_hms(elapsed_seconds):
    if elapsed_seconds is None or is_nan_metric(elapsed_seconds):
        return ""

    total_seconds = max(0, int(round(float(elapsed_seconds))))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_epoch_postfix(mean_train_loss, val_loader, epoch_metrics, early_stopper, dist_edge_stats):
    epoch_postfix = {"train_loss": f"{mean_train_loss:.6f}"}
    if val_loader is not None:
        epoch_postfix["val_loss"] = f'{float(epoch_metrics["val_loss"]):.6f}'
        epoch_postfix["val_dice"] = f'{float(epoch_metrics["dice"]):.6f}'
        if early_stopper is not None and not is_nan_metric(early_stopper.best_val_dice):
            epoch_postfix["best_val_dice"] = f"{float(early_stopper.best_val_dice):.6f}"
        else:
            epoch_postfix["best_val_dice"] = "nan"
        if early_stopper is not None and early_stopper.enabled:
            epoch_postfix["es"] = f"{int(early_stopper.counter)}/{int(early_stopper.patience)}"
        else:
            epoch_postfix["es"] = "disabled"
    elif dist_edge_stats is not None:
        epoch_postfix["edge_mean"] = f'{float(dist_edge_stats["edge_mean"]):.6f}'
        epoch_postfix["edge_nonzero"] = f'{float(dist_edge_stats["edge_nonzero_ratio"]):.6f}'
    return epoch_postfix


class EarlyStopping:
    def __init__(
        self,
        monitor_metric="val_dice",
        mode="max",
        patience=20,
        min_delta=0.001,
        tie_break_with_loss=True,
        tie_eps=1e-4,
        stop_interval=10,
    ):
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.tie_break_with_loss = bool(tie_break_with_loss)
        self.tie_eps = float(tie_eps)
        self.stop_interval = max(1, int(stop_interval))

        self.enabled = self.patience > 0
        self.best_monitor = float("nan")
        self.best_val_dice = float("nan")
        self.best_val_loss = float("nan")
        self.best_epoch = 0
        self.counter = 0
        self.pending_stop = False

    def _is_improved(self, monitor_value):
        if self.best_epoch == 0:
            return not is_nan_metric(monitor_value)
        if is_nan_metric(monitor_value):
            return False
        if self.mode == "max":
            return monitor_value > (self.best_monitor + self.min_delta)
        return monitor_value < (self.best_monitor - self.min_delta)

    def _is_tie_break_improved(self, val_dice, val_loss):
        if self.monitor_metric != "val_dice":
            return False
        if not self.tie_break_with_loss:
            return False
        if self.best_epoch == 0:
            return False
        if is_nan_metric(val_dice) or is_nan_metric(val_loss) or is_nan_metric(self.best_val_loss):
            return False
        if abs(float(val_dice) - float(self.best_val_dice)) > self.tie_eps:
            return False
        return float(val_loss) < float(self.best_val_loss)

    def step(self, monitor_value, val_dice, val_loss, epoch):
        improved = self._is_improved(monitor_value)
        used_tie_break = False
        if not improved and self._is_tie_break_improved(val_dice, val_loss):
            improved = True
            used_tie_break = True

        if improved:
            self.best_monitor = float(monitor_value)
            self.best_val_dice = float(val_dice)
            self.best_val_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.counter = 0
            self.pending_stop = False
        else:
            if self.enabled:
                self.counter += 1
                if self.counter >= self.patience:
                    self.pending_stop = True

        should_stop = bool(self.enabled and self.pending_stop and (int(epoch) % self.stop_interval == 0))
        waiting_for_boundary = bool(self.enabled and self.pending_stop and not should_stop)

        return {
            "improved": improved,
            "used_tie_break": used_tie_break,
            "counter": int(self.counter),
            "best_monitor": float(self.best_monitor),
            "best_val_dice": float(self.best_val_dice),
            "best_val_loss": float(self.best_val_loss),
            "best_epoch": int(self.best_epoch),
            "should_stop": should_stop,
            "waiting_for_boundary": waiting_for_boundary,
        }
