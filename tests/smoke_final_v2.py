"""CPU smoke test for the final Stage 2-4 architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.pooling import RationalePooling
from models.re_module import RationableExtraction
from models.td_head import TDHead


def main() -> None:
    batch_size = 2
    hidden = 32
    num_p = 5
    num_d = 4

    stage1 = {
        "H_u": torch.randn(
            batch_size,
            hidden,
            requires_grad=True,
        ),
        "hP_cond": torch.randn(
            num_p,
            hidden,
            requires_grad=True,
        ),
        "hD_cond": torch.randn(
            num_d,
            hidden,
            requires_grad=True,
        ),
    }
    batch = {
        "U_input_ids": torch.ones(
            batch_size,
            3,
            dtype=torch.long,
        ),
        "sample_map_P": torch.tensor([0, 0, 1, 1, 1]),
        "sample_map_D": torch.tensor([0, 1, 1, 1]),
        "R_P": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0]),
        "R_D": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "T": torch.tensor([1.0, 0.0]),
    }

    stage2 = RationableExtraction(
        hidden,
        use_task_adapter=True,
    )
    stage3 = RationalePooling(
        hidden,
        topk_claims=3,
        use_task_adapter=True,
        detach_rationale_for_tp=True,
        mix_gate_init=-1.5,
    )
    stage4 = TDHead(
        hidden,
        num_heads=4,
        use_global_residual=True,
        rationale_scale_init=-1.5,
    )

    s2 = stage2(stage1)
    s3 = stage3(stage1, s2, batch)

    stage4.eval()
    rationale = stage4(
        stage1,
        s3,
        input_mode="rationale",
    )
    global_only = stage4(
        stage1,
        None,
        input_mode="global_only",
    )

    # Zero-initialized residual means phase 2 starts from phase-1 predictions.
    assert torch.allclose(
        rationale["T_logit"],
        global_only["T_logit"],
    )

    loss = (
        rationale["T_logit"].sum()
        + s2["logits_P"].sum()
        + s2["logits_D"].sum()
    )
    loss.backward()

    test_hidden = torch.randn(3, hidden)
    assert torch.equal(
        stage2.re_adapter(test_hidden),
        test_hidden,
    )
    assert torch.equal(
        stage3.tp_adapter(test_hidden),
        test_hidden,
    )

    print("Final V2 smoke test passed")


if __name__ == "__main__":
    main()
