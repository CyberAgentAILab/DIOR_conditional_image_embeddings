import torch
from pytorch_metric_learning.utils.accuracy_calculator import (
    AccuracyCalculator,
    get_relevance_mask,
    maybe_get_avg_of_avgs,
    nan_accuracy,
    try_getting_not_lone_labels,
)


def recall_at_k(
    knn_labels,
    gt_labels,
    k,
    ref_includes_query,
    label_counts,
    avg_of_avgs,
    return_per_class,
    label_comparison_fn,
):
    curr_knn_labels = knn_labels[:, :k]
    same_label = label_comparison_fn(gt_labels, curr_knn_labels)
    retrieved_relevant = torch.sum(same_label, dim=1).type(torch.float64)

    relevance_mask, count_per_query = get_relevance_mask(
        knn_labels.shape[:2],
        gt_labels,
        ref_includes_query,
        label_counts,
    )
    total_relevant = count_per_query.type(torch.float64)

    non_zero_mask = total_relevant > 0
    recall_per_sample = torch.zeros_like(retrieved_relevant)
    recall_per_sample[non_zero_mask] = retrieved_relevant[non_zero_mask] / total_relevant[non_zero_mask]

    return maybe_get_avg_of_avgs(recall_per_sample, gt_labels, avg_of_avgs, return_per_class)


class ExtendedAccuracyCalculator(AccuracyCalculator):
    def calculate_recall_at_1(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        ref_includes_query,
        label_counts,
        **kwargs,
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(knn_labels, query_labels, not_lone_query_mask)
        if knn_labels is None:
            return nan_accuracy(label_counts[0], self.return_per_class)

        return recall_at_k(
            knn_labels,
            query_labels[:, None],
            1,
            ref_includes_query,
            label_counts,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn,
        )
