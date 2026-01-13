from typing import Any, Dict, Tuple, cast

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassConfusionMatrix, MultilabelAccuracy

from src.utils.conf_mat import plot_confusion_matrix
from src.utils.to_class import to_one_integer_class


class InstrumentCNNModule(LightningModule):
    """Detects in each window the presence of each audio source class.

    Input shape: (B, n_mels, n_frame)
    Output shape: (B, n_classes)

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        classifier: torch.nn.Module,
        optimizer: torch.optim.Optimizer,  # type: ignore
        scheduler: torch.optim.lr_scheduler,  # type: ignore
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.feature_extractor = feature_extractor
        self.classifier = classifier

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = MultilabelAccuracy(num_labels=3)
        self.val_acc = MultilabelAccuracy(num_labels=3)
        self.test_acc = MultilabelAccuracy(num_labels=3)

        # metric for testing analysis : ConfusionMatrix
        self.test_confmat = MulticlassConfusionMatrix(num_classes=8)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the feature extractor and the classifier.

        :param x: A tensor of spectrum windows. Shape (B, n_mels, n_frame)
        :return: A tensor of logits. Shape (B, 3)
        """
        hidden_features = self.feature_extractor(x)
        return self.classifier(hidden_features)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, NDArray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, NDArray]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions. Bool Tensor of shape (B, 3).
            - A tensor of target labels.
        """
        x, y, source_positions = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits) >= 0.5
        return loss, preds, y, source_positions

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, NDArray], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, NDArray], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, _ = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, NDArray], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.test_confmat.update(to_one_integer_class(preds), to_one_integer_class(targets))

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, NDArray], batch_idx: int
    ) -> Tuple[torch.Tensor, NDArray]:
        """The complete predict step."""
        x, _, source_positions = batch
        preds = torch.sigmoid(self(x)) >= 0.5
        return preds, source_positions

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # compute the confusion matrix
        confmat = self.test_confmat.compute()
        logger = cast(MLFlowLogger, self.logger)
        if logger.run_id is not None:
            fig = plot_confusion_matrix(confmat, [f"{n:03b}" for n in range(8)])
            cast(MlflowClient, logger.experiment).log_figure(
                run_id=logger.run_id,
                figure=fig,
                artifact_file="plots/confusion_matrix.png",
            )
        self.test_confmat.reset()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        dummy = torch.zeros(1, 1, 64, 22)  # dummy mel spectrum
        self.forward(dummy)
        if self.hparams.compile and stage == "fit":
            self.feature_extractor = torch.compile(self.feature_extractor)
            self.classifier = torch.compile(self.classifier)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = InstrumentCNNModule(None, None, None, None, None)
