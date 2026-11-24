import numpy as np
import torch

from src.generation_module.MLM import BOS_TOKEN_ID, MLM
from src.loss.PGS_loss import LayerAlignmentLoss
from src.open_clip.transformer import Transformer
from src.training.metrics import get_metrics


class _Args:
    token_txt = 64
    s_context_length = 16
    s_vocab_size = 50000


def test_mlm_batched_matches_per_sample():
    torch.manual_seed(0)
    mlm = MLM(_Args())
    mlm.eval()

    batch, length = 4, 16
    encode_output = torch.randn(batch, length, _Args.token_txt)
    lengths = torch.tensor([1, 3, 5, 7])

    batched = mlm(encode_output, lengths)

    reference = []
    for i in range(batch):
        state = mlm.decoder.init_state(encode_output[i], lengths[i])
        seq = [BOS_TOKEN_ID]
        step_logits = []
        for j in range(int(lengths[i])):
            current = torch.tensor([seq[-1]])
            logits = mlm.decoder(current, state, j)
            step_logits.append(logits)
            seq.append(logits[:, -1, :].argmax(dim=-1).item())
        reference.append(mlm.pad_logits(torch.cat(step_logits, dim=1), mlm.max_logits_len))
    reference = torch.cat(reference, dim=0)

    assert batched.shape == (batch, mlm.max_logits_len, _Args.s_vocab_size)
    assert torch.allclose(batched, reference, atol=1e-4)
    for i in range(batch):
        n = int(lengths[i])
        assert batched[i, :n].abs().sum() > 0
        assert batched[i, n:].abs().sum() == 0


def test_encoder_mask_supports_batched_lengths():
    attention = MLM(_Args()).decoder.layers[0].attention2
    mask = attention.build_encoder_mask(torch.tensor([1, 3, 8]), 8, "cpu")
    assert mask.shape == (3, 1, 1, 8)
    assert mask[0, 0, 0].tolist() == [1., 0., 0., 0., 0., 0., 0., 0.]
    assert mask[1, 0, 0].tolist() == [1., 1., 1., 0., 0., 0., 0., 0.]
    assert mask[2, 0, 0].tolist() == [1.] * 8


def test_pgs_projection_matches_loop_reference():
    torch.manual_seed(1)
    for layer in (0, 6, 11):
        criterion = LayerAlignmentLoss(layer)
        student = torch.randn(7, 16)
        teacher = torch.randn(7, 16)

        angle, distance = 0.0, 0.0
        for s, t in zip(student, teacher):
            s_norm = s / s.norm(p=2)
            t_norm = t / t.norm(p=2)
            angle = angle + 1 - torch.dot(s_norm, t_norm)
            distance = distance + torch.nn.functional.mse_loss(s, t)
        expected = (angle * (1 - criterion.distance_weight) +
                    distance * criterion.distance_weight) / len(student)

        assert torch.allclose(criterion(student, teacher), expected, atol=1e-5)


def test_transformer_collects_only_requested_layers():
    torch.manual_seed(2)
    transformer = Transformer(width=32, layers=4, heads=4)
    transformer.eval()
    x = torch.randn(5, 2, 32, requires_grad=True)

    out_full, inter_full, attn_full = transformer(x)
    out_sparse, inter_sparse, attn_sparse = transformer(x, layers=(0, 2), attn_layers=(1,))

    assert torch.allclose(out_full, out_sparse, atol=1e-5)
    assert [item is not None for item in inter_sparse] == [True, False, True, False]
    assert [item is None for item in attn_sparse] == [True, False, True, True]
    assert torch.allclose(attn_full[1], attn_sparse[1], atol=1e-5)

    out_sparse.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_get_metrics_matches_full_matrix():
    torch.manual_seed(3)
    num_samples, dim = 37, 16
    image_features = torch.randn(num_samples, dim)
    text_features = torch.randn(num_samples, dim)
    logit_scale = torch.tensor(2.5)

    actual = get_metrics(image_features, text_features, logit_scale, chunk_size=8)

    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t()
    expected = {}
    ground_truth = torch.arange(num_samples).view(-1, 1)
    for name, logits in {"image_to_text": logits_per_image, "text_to_image": logits_per_text}.items():
        preds = torch.where(torch.argsort(logits, descending=True) == ground_truth)[1].numpy()
        expected[f"{name}_mean_rank"] = preds.mean() + 1
        expected[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in (1, 5, 10):
            expected[f"{name}_R@{k}"] = np.mean(preds < k)

    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert np.isclose(actual[key], value)

    empty = get_metrics(torch.empty(0, dim), torch.empty(0, dim), logit_scale)
    assert empty == {}
