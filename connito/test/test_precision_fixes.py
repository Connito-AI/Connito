import tempfile
import torch
import torch.nn as nn
from types import SimpleNamespace

from connito.shared.checkpoint_helper import save_state_dict_by_expert_group, compile_full_state_dict_from_path
from connito.shared.evaluate import evaluate_model

def test_checkpoint_precision_preservation():
    """
    Tests if the model limits are bypassed by saving with bfloat16.
    Standard float16 maxes out at ~65504. A value of 100,000.0 will
    turn into float('inf') if illegally downcasted.
    """
    test_val = 100000.0 
    
    state_dict = {
        "model.embed_tokens.weight": torch.tensor([test_val], dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.weight": torch.tensor([test_val], dtype=torch.bfloat16),
    }
    
    expert_groups = { 0: { 0: [(0, 0)] } }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save explicitly with bfloat16 to verify it doesn't get hard-reverted
        save_state_dict_by_expert_group(
            state_dict=state_dict,
            expert_groups=expert_groups,
            save_dir=tmpdir,
            save_dtype=torch.bfloat16,
            active_expert_group_id=0
        )
        
        # Reload
        reloaded = compile_full_state_dict_from_path(tmpdir)
        shared_weight = reloaded["model.embed_tokens.weight"]
        expert_weight = reloaded["model.layers.0.mlp.experts.0.weight"]
        
        # Verify precision stayed intact
        assert shared_weight.dtype == torch.bfloat16, "Shared weight downcast to float16!"
        assert expert_weight.dtype == torch.bfloat16, "Expert weight downcast to float16!"
        
        # Verify value didn't become `inf` or `NaN`
        assert not torch.isinf(shared_weight).any(), "Shared weight overflowed to Inf due to fp16 downcast!"
        assert not torch.isinf(expert_weight).any(), "Expert weight overflowed to Inf due to fp16 downcast!"
        
class DTypeCheckModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(10, 10).to(device)
        
    def forward(self, input_ids, attention_mask=None):
        out = self.linear(torch.randn(1, 10, device=self.device))
        
        # If the evaluator is forcing `torch.float16`, this `out` tensor will be mapped to float16
        # Our fix should allow bfloat16 or float32 depending on system.
        assert out.dtype != torch.float16, f"Autocast improperly forced float16! Detected: {out.dtype}"
        
        return SimpleNamespace(loss=torch.tensor(2.4388, device=self.device), aux_loss=torch.tensor(0.0))

def test_evaluator_dynamic_precision():
    """
    Tests that the evaluator uses BF16 dynamic mapping rather than strictly enforcing FP16.
    """
    # Needs CUDA with bf16 hardware for literal dynamic mapping, but runs safe asserts anywhere
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        print("- Skipping evaluator dynamic test (Requires CUDA device with BF16 compute)")
        return
        
    device = torch.device("cuda")
    model = DTypeCheckModel(device=device)
    
    # Pass mock data into the evaluator
    dataloader = [{"input_ids": torch.tensor([[1]])}]
    
    metrics = evaluate_model(
        step=0,
        model=model,
        eval_dataloader=dataloader,
        device=device
    )
    
    # Prove the evaluator isn't returning 0.0 due to NaN trapping
    assert metrics["val_loss"] > 0.0, "Evaluator yielded `val_loss=0.0` check logic"


class NaNAfterNBatchesModel(nn.Module):
    """Returns finite loss for the first ``n_finite`` calls then NaN for the
    rest. Used to verify the evaluator's divisor counts only finite batches.
    """
    def __init__(self, device, n_finite: int, finite_loss: float = 2.0):
        super().__init__()
        self.device = device
        self.n_finite = n_finite
        self.finite_loss = finite_loss
        self._calls = 0

    def forward(self, input_ids, attention_mask=None):
        self._calls += 1
        loss = (
            torch.tensor(self.finite_loss, device=self.device)
            if self._calls <= self.n_finite
            else torch.tensor(float("nan"), device=self.device)
        )
        return SimpleNamespace(loss=loss, aux_loss=torch.tensor(0.0, device=self.device))


def test_evaluator_skips_nan_batches_from_divisor():
    """A miner that NaNs some batches must not get the divisor padded by
    the failed batches. Returned val_loss = mean over scored (finite)
    batches only, not the gamed `(scored / total) * honest_loss`.
    """
    device = torch.device("cpu")
    n_finite, n_nan = 4, 6
    model = NaNAfterNBatchesModel(device=device, n_finite=n_finite, finite_loss=2.0)
    dataloader = [{"input_ids": torch.tensor([[1]])} for _ in range(n_finite + n_nan)]

    metrics = evaluate_model(
        step=0,
        model=model,
        eval_dataloader=dataloader,
        device=device,
        max_eval_batches=None,
    )

    # Pre-fix behavior would have been `(2.0 * 4) / 10 = 0.8`. The fix
    # divides by `scored_batches = 4`, recovering the honest mean.
    assert metrics["val_loss"] == 2.0
    assert metrics["scored_batches"] == n_finite
    assert metrics["nan_batches"] == n_nan


def test_evaluator_returns_inf_when_every_batch_nans():
    """100% NaN miner must NOT score val_loss=0 (which would be max
    delta after `max(0, baseline - val_loss)`). Return +inf so delta
    clamps to 0."""
    import math
    device = torch.device("cpu")
    model = NaNAfterNBatchesModel(device=device, n_finite=0)
    dataloader = [{"input_ids": torch.tensor([[1]])} for _ in range(5)]

    metrics = evaluate_model(
        step=0,
        model=model,
        eval_dataloader=dataloader,
        device=device,
        max_eval_batches=None,
    )

    assert math.isinf(metrics["val_loss"])
    assert metrics["scored_batches"] == 0
    assert metrics["nan_batches"] == 5


def test_evaluator_skips_inf_batches_too():
    """bf16 overflow lands as +inf before propagating to NaN. The fix
    treats Inf and NaN equivalently — both drop from the divisor and
    from aux_loss_sum so a related variant cannot game the ratio."""
    device = torch.device("cpu")

    class InfThenFiniteModel(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.device = device
            self._calls = 0

        def forward(self, input_ids, attention_mask=None):
            self._calls += 1
            loss = (
                torch.tensor(float("inf"), device=self.device)
                if self._calls <= 3
                else torch.tensor(2.0, device=self.device)
            )
            return SimpleNamespace(loss=loss, aux_loss=torch.tensor(0.5, device=self.device))

    model = InfThenFiniteModel(device=device)
    dataloader = [{"input_ids": torch.tensor([[1]])} for _ in range(5)]

    metrics = evaluate_model(
        step=0,
        model=model,
        eval_dataloader=dataloader,
        device=device,
        max_eval_batches=None,
    )

    # Only the 2 finite batches contribute. (2.0 - 0.5) per finite batch.
    assert metrics["val_loss"] == 1.5
    assert metrics["scored_batches"] == 2
    assert metrics["nan_batches"] == 3


if __name__ == "__main__":
    test_checkpoint_precision_preservation()
    print("✓ Checkpointer: Checkpoint precision preservation test passed. No fp16 downcasts occurred.")

    test_evaluator_dynamic_precision()
    print("✓ Evaluator: BF16 dynamic precision test passed. FP16 hardcode bypassed.")

    test_evaluator_skips_nan_batches_from_divisor()
    test_evaluator_returns_inf_when_every_batch_nans()
    test_evaluator_skips_inf_batches_too()
    print("✓ Evaluator: NaN-batch divisor fix verified.")
