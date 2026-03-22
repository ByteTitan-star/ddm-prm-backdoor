import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        """Wrap a base loss with optional knowledge distillation from a teacher.

        Args:
            base_criterion (torch.nn.Module): Primary loss (e.g. cross-entropy) on student logits.
            teacher_model (torch.nn.Module): Frozen teacher producing reference logits.
            distillation_type (str): One of ``'none'``, ``'soft'`` (KL), or ``'hard'`` (CE on teacher preds).
            alpha (float): Weight of distillation term; base loss weight is ``1 - alpha``.
            tau (float): Temperature for soft distillation.
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """Compute total loss from student outputs and optional KD logits.

        Args:
            inputs (Tensor): Inputs fed to the teacher when distillation is enabled.
            outputs (Tensor or tuple): Student logits, or ``(logits, kd_logits)`` if KD branch exists.
            labels (Tensor): Ground-truth labels for ``base_criterion``.

        Returns:
            Tensor: Scalar loss.
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
