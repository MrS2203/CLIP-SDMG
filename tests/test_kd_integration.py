import torch

from src.loss.common import DAB_LAYERS, DISTILL_LAYERS
from src.open_clip import create_model, get_tokenizer
from src.open_clip.loss import KDClipLoss


class _Args:
    model = "ViT-T-16"
    t_model = "ViT-B-16"
    s_embed_dim = 512
    t_embed_dim = 512
    token_txt = 384
    t_token_txt = 512
    s_vocab_size = 49408
    s_context_length = 77
    epochs = 32
    alpha_fd_loss = 1.0
    alpha_pgs_loss = 1.0
    alpha_dab_loss = 1.0
    alpha_gen_loss = 1.0


def test_kd_forward_and_losses_are_finite():
    torch.manual_seed(0)
    student = create_model("ViT-T-16", pretrained=None, precision="fp32", device="cpu")
    teacher = create_model("ViT-B-16", pretrained=None, precision="fp32", device="cpu")
    student.eval()
    teacher.eval()

    tokenizer = get_tokenizer("ViT-T-16")
    texts = tokenizer(["a photo of a cat", "a dog and a cat"])
    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = student(images, texts, distill=True,
                          layers=DISTILL_LAYERS, attn_layers=DAB_LAYERS)
        teacher_outputs = teacher(images, texts,
                                  layers=DISTILL_LAYERS, attn_layers=DAB_LAYERS)

    image_features, intermediates, text_features, txt_intermediates, logit_scale, attention = outputs
    t_image_features, t_intermediates, t_text_features, t_txt_intermediates, _, t_attention = teacher_outputs

    criterion = KDClipLoss(_Args(), cache_labels=True)
    criterion.eval()
    losses = criterion(
        image_features, text_features, logit_scale,
        t_image_features, t_text_features,
        intermediates, t_intermediates,
        txt_intermediates, t_txt_intermediates,
        attention, t_attention, texts, epoch=1)

    values = [losses.task_loss, losses.fd_loss, losses.gen_loss, losses.pgs_loss, losses.dab_loss]
    assert all(torch.isfinite(value) for value in values)
    assert losses.pgs_loss.item() > 0
    assert losses.dab_loss.item() > 0
    assert losses.gen_loss.item() > 0
