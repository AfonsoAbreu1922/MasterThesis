from torchmetrics.functional.classification import (
    binary_auroc,
    binary_accuracy,
    binary_precision,
    binary_recall,
    binary_specificity
)


class FixMatchPerformanceMetrics:
    @staticmethod
    def get(logits, labels, subset_type_prefix):
        preds = logits['from_non_augmented_labeled_images'].argmax(1)
        target = labels.squeeze(1)

        performance_metrics = {
            f"{subset_type_prefix}_auroc": binary_auroc(
                preds=preds.float(),
                target=target.int()
            ).item(),
            f"{subset_type_prefix}_accuracy": binary_accuracy(
                preds=preds,
                target=target
            ).item(),
            f"{subset_type_prefix}_precision": binary_precision(
                preds=preds,
                target=target
            ).item(),
            f"{subset_type_prefix}_sensitivity": binary_recall(
                preds=preds,
                target=target
            ).item(),
            f"{subset_type_prefix}_specificity": binary_specificity(
                preds=preds,
                target=target
            ).item()
        }
        return performance_metrics