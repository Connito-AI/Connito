import time

from connito.shared.telemetry import TelemetryManager, SystemStatePoller
from connito.shared.telemetry import (
    MINER_TRAINING_LOSS, VALIDATOR_MINER_SCORE, GPU_VRAM_ALLOCATED_BYTES, 
    GPU_UTILIZATION_PERCENT, MINER_LEARNING_RATE, VALIDATOR_EVAL_LOSS, 
    MOE_EXPERT_LOAD, MINER_TOKENS_PER_SEC
)
import random

def run_mock():
    print("Starting Mock Node on 8120")
    TelemetryManager().start_server(8120)

    class MockSubtensor:
        def get_current_block(self):
            return 50000

    class MockPhaseManager:
        def get_phase(self, block):
            class PhaseResp:
                phase_index = 2
                blocks_remaining_in_phase = 14
            return PhaseResp()
            
    poller = SystemStatePoller(
        subtensor=MockSubtensor(),
        phase_manager=MockPhaseManager(),
        interval_sec=2
    )
    poller.start()
    
    print("Generating simulated metrics...")
    try:
        cur_loss = 3.5
        for i in range(1000):
            # Training stuff
            cur_loss = cur_loss * 0.98 + random.uniform(-0.05, 0.05)
            MINER_TRAINING_LOSS.labels(expert_group="math").set(cur_loss)
            MINER_TRAINING_LOSS.labels(expert_group="logic").set(cur_loss * 1.2)
            MINER_LEARNING_RATE.set(0.001 * (0.99 ** i))
            MINER_TOKENS_PER_SEC.set(2400 + random.uniform(-100, 100))
            
            # Validation stuff
            VALIDATOR_EVAL_LOSS.labels(expert_group="math").set(cur_loss + 0.1)
            VALIDATOR_MINER_SCORE.labels(miner_uid="10").set(0.85 + random.uniform(-0.02, 0.02))
            VALIDATOR_MINER_SCORE.labels(miner_uid="11").set(0.0) # Bad miner
            
            # System / VRAM
            GPU_VRAM_ALLOCATED_BYTES.labels(device="0").set(40 * 1024**3 + random.uniform(-1024**2, 1024**2)) # ~40GB
            GPU_UTILIZATION_PERCENT.labels(device="0").set(random.uniform(90, 100))
            
            # MoE Routing Load
            for layer in range(2):
                for e in range(8):
                    MOE_EXPERT_LOAD.labels(layer_idx=str(layer), expert_idx=str(e)).set(random.uniform(100, 500))

            time.sleep(2)
            print(f"Step {i} | Simulating dashboard load...")
    except KeyboardInterrupt:
        pass
    finally:
        poller.stop()

if __name__ == "__main__":
    run_mock()
