import torch
import torch.nn.functional as F

class EMASmooth:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value

    def get(self):
        return self.value if self.value is not None else 0.0

def feature_distillation_loss(
    teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
    teacher_feat = F.normalize(teacher_feat, dim=-1)
    student_feat = F.normalize(student_feat, dim=-1)
    cosine_sim = (teacher_feat * student_feat).sum(dim=-1)
    return (1 - cosine_sim).mean()

def cross_modal_distillation_loss(logit_scale: torch.Tensor,
                                  student_img_feat: torch.Tensor,
                                  student_text_feat: torch.Tensor,
                                  teacher_img_feat: torch.Tensor,
                                  teacher_text_feat: torch.Tensor,
                                  temperature = 2.0):
    with torch.no_grad():
        teacher_logits_per_image = logit_scale.exp() * (teacher_img_feat @ teacher_text_feat.t())
        teacher_probs = torch.softmax(teacher_logits_per_image / temperature, dim=-1)

    student_logits_per_image = logit_scale.exp() * (student_img_feat @ student_text_feat.t())
    loss = F.kl_div(
        input=torch.log_softmax(student_logits_per_image / temperature, dim=-1),
        target=teacher_probs,
        reduction="batchmean"
    ) * (temperature ** 2)
    return loss

