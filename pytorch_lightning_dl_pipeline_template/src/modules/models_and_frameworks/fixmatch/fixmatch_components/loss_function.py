from torch.nn.functional import binary_cross_entropy_with_logits, one_hot
import torch
import torch.nn.functional as F

from src.modules.data.metadataframe.preprocessed_metadataframe \
    import PreprocessedMetadataFrame


class FixMatchLossFunction(torch.nn.Module):
    def __init__(self, config, experiment_execution_paths):
        super().__init__()
        self.config = config

        self.classification_loss_type = self.config.classification_loss_type
        self.class_weights = \
            self._get_class_weights(experiment_execution_paths)
        self.lambda_u = self.config.unsupervised_loss_relative_weight
        self.tau = self.config.pseudo_label_approval_threshold
        self.eta = self.config.eta
        self.iota = self.config.iota

    def forward(self, logits, labels, pipeline_step="training"):
        if 'from_weakly_augmented_unlabeled_images' not in logits:
            print("Logits does not contain 'from_weakly_augmented_unlabeled_images'")

        print("\nFixMatchLossFunction.forward")
        supervised_loss = None
        if logits['from_weakly_augmented_labeled_images'] is not None:
            supervised_loss = self._get_supervised_loss(
                logits['from_weakly_augmented_labeled_images'],
                labels
            )

        if pipeline_step == "training":
            unsupervised_loss = None
            if logits['from_weakly_augmented_unlabeled_images'] is not None:
                unsupervised_loss = self._get_unsupervised_loss(
                    logits['from_weakly_augmented_unlabeled_images'],
                    logits['from_strongly_augmented_unlabeled_images']
                )

            loss = None
            if supervised_loss is not None or unsupervised_loss is not None:
                if supervised_loss is not None and unsupervised_loss is None:
                    loss = supervised_loss
                    print(f"loss = {supervised_loss} = {loss}")
                elif supervised_loss is None and unsupervised_loss is not None:
                    loss = self.lambda_u * unsupervised_loss
                    print(f"loss = {self.lambda_u} * {unsupervised_loss} = {loss}")
                else:
                    loss = supervised_loss + self.lambda_u * unsupervised_loss
                    print(
                        f"loss = {supervised_loss} + {self.lambda_u} * {unsupervised_loss} = {loss}")
                print(loss)
                loss = loss.clamp(min=0.2)

            unlabeled_data_utilization = self._get_unlabeled_data_utilization(
                logits['from_weakly_augmented_unlabeled_images']
            )

            return loss, unlabeled_data_utilization
        else:
            loss = supervised_loss
            print(f"loss = {supervised_loss} = {loss}")
            return loss

    @staticmethod
    def _compute_dice_loss(logits, ground_truth_labels, epsilon = 1e-6):
        predicted_class_probabilities = torch.sigmoid(logits)
        intersection = (
            predicted_class_probabilities * ground_truth_labels
        ).sum()
        union = predicted_class_probabilities.sum() + ground_truth_labels.sum()
        dice_losses = 1 - (2. * intersection + epsilon) / (union + epsilon)
        dice_loss = dice_losses.mean()
        return dice_loss

    def _compute_focal_loss(self, logits, ground_truth_labels):
        bce_losses = F.binary_cross_entropy_with_logits(
            input=logits,
            target=ground_truth_labels,
            reduction='none'
        )
        predicted_class_probabilities = torch.sigmoid(logits)
        true_class_predicted_probabilities = torch.where(
            ground_truth_labels == 1.0,
            predicted_class_probabilities,
            1 - predicted_class_probabilities
        )
        label_weights = torch.where(
            ground_truth_labels == 1.0,
            self.class_weights[1],
            self.class_weights[0]
        )
        focal_losses = (
            (1 - true_class_predicted_probabilities) ** self.iota
            * bce_losses * label_weights
        )
        focal_loss = focal_losses.mean()
        return focal_loss

    def _get_supervised_loss(
            self,
            logits_from_weakly_augmented_labeled_images,
            labels
    ):
        logits = logits_from_weakly_augmented_labeled_images
        ground_truth_labels = one_hot(labels.squeeze().long(), num_classes=2).float()

        if self.classification_loss_type == "BCE":
            bce_loss = binary_cross_entropy_with_logits(
                input=logits,
                target=ground_truth_labels,
                weight=self.class_weights
            )
            supervised_loss = bce_loss
        elif self.classification_loss_type == "BCE + Dice":
            bce_loss = binary_cross_entropy_with_logits(
                input=logits,
                target=ground_truth_labels,
                weight=self.class_weights
            )
            dice_loss = \
                self._compute_dice_loss(logits, ground_truth_labels)
            supervised_loss = bce_loss + self.eta * dice_loss
        elif self.classification_loss_type == "Focal":
            focal_loss = \
                self._compute_focal_loss(logits, ground_truth_labels)
            supervised_loss = focal_loss
        else:
            raise ValueError(
                f"Invalid classification loss type: "
                f"{self.classification_loss_type}. "
                f"Supported types are 'BCE', 'BCE + Dice' and 'Focal'"
            )
        return supervised_loss

    def _get_unlabeled_data_utilization(
            self,
            logits_from_weakly_augmented_unlabeled_images
    ):
        if logits_from_weakly_augmented_unlabeled_images is None:
            return torch.tensor(0.0) # ou outro valor neutro conforme o contexto

        predicted_label_probabilities = torch.nn.functional.softmax(
            logits_from_weakly_augmented_unlabeled_images, dim=1
        )

        predicted_label_approval_filter = torch.any(
            predicted_label_probabilities >= self.tau, 1
        )
        unlabeled_data_utilization = (
            predicted_label_approval_filter.sum().item()
            / predicted_label_probabilities.shape[0]
        )
        return unlabeled_data_utilization

    def _get_unsupervised_loss(
            self,
            logits_from_weakly_augmented_unlabeled_images,
            logits_from_strongly_augmented_unlabeled_images
    ):
        predicted_label_probabilities = torch.nn.functional.softmax(
            logits_from_weakly_augmented_unlabeled_images,
            dim=1
        )
        predicted_label_approval_filter = torch.any(
            predicted_label_probabilities >= self.tau, 1
        )
        print(f"{predicted_label_approval_filter.sum() = }")
        if torch.any(predicted_label_approval_filter):
            pseudo_labels = one_hot(
                logits_from_weakly_augmented_unlabeled_images.argmax(dim=1),
                num_classes=2
            ).float()
            unsupervised_loss = binary_cross_entropy_with_logits(
                input=logits_from_strongly_augmented_unlabeled_images[
                    predicted_label_approval_filter
                ],
                target=pseudo_labels[predicted_label_approval_filter],
                reduction="sum"
            ) / pseudo_labels.shape[0]
        else:
            unsupervised_loss = torch.tensor(
                data=0.0,
                device=self.config.device,
                requires_grad=True
            )
        return unsupervised_loss

    def _get_class_weights(self, experiment_execution_paths):
        if self.config.apply_weights:
            metadataframe = PreprocessedMetadataFrame(
                config=self.config.metadataframe
                    [self.config.labeled_dataset_name],
                preprocessed_data_dir_path=
                    experiment_execution_paths.preprocessed_data_dir_path
                        [self.config.labeled_dataset_name]
            )
            lung_nodule_metadataframe = \
                metadataframe.get_lung_nodule_metadataframe()
            class_counts = lung_nodule_metadataframe[
                f'label_{self.config.supervised_loss_label}'
            ].value_counts().sort_index()
            class_weights = torch.tensor(
                (class_counts.min() / class_counts).tolist()
            ).to(self.config.device)
        else:
            class_weights = torch.tensor([1.0, 1.0]).to(self.config.device)
        return class_weights