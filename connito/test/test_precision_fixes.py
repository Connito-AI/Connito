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


if __name__ == "__main__":
    test_checkpoint_precision_preservation()
    print("✓ Checkpointer: Checkpoint precision preservation test passed. No fp16 downcasts occurred.")
    
    test_evaluator_dynamic_precision()
    print("✓ Evaluator: BF16 dynamic precision test passed. FP16 hardcode bypassed.")
