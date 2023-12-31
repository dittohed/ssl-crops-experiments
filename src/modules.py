import copy
import torch

from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad


# TODO: doublecheck
class DINO(torch.nn.Module):
    """
    Copy-paste from https://docs.lightly.ai/self-supervised-learning/examples/dino.html.
    """

    def __init__(self, backbone, input_dim):
        super().__init__()

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, hidden_dim=1024, output_dim=1024, freeze_last_layer=1
        )
        
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim, hidden_dim=1024, output_dim=1024
        )

        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)

        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        
        return z