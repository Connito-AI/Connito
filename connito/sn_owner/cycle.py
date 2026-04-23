import bittensor

from connito.shared.config import CycleCfg
from connito.shared.cycle import PhaseNames, PhaseResponse



class PhaseManager:
    def __init__(self, cycle_cfg: CycleCfg, subtensor: bittensor.Subtensor):
        self.cycle_cfg = cycle_cfg
        self.subtensor = subtensor
        self.phases = self.init_phases(cycle_cfg, PhaseNames())
        self.cycle_length = sum(p["length"] for p in self.phases)

    @classmethod
    def from_dict(cls, cycle_dict: dict, subtensor: bittensor.Subtensor) -> "PhaseManager":
        """Build from a dict matching CycleCfg (e.g. parsed from the owner's /get_cycle_config JSON)."""
        return cls(CycleCfg(**cycle_dict), subtensor)

    def init_phases(self, cycle_cfg: CycleCfg, names):
        # ordered phase
        phases = [
            {"name": names.distribute, "length": cycle_cfg.distribute_period},  # miner download model from validator
            {"name": names.train, "length": cycle_cfg.train_period},  # miner train
            {
                "name": names.miner_commit_1,
                "length": cycle_cfg.commit_period,
            },  # miner commit model hash and validator commit seed
            {
                "name": names.miner_commit_2,
                "length": cycle_cfg.commit_period,
            },  # miner commit model hash and validator commit seed
            {"name": names.submission, "length": cycle_cfg.submission_period},  # miner submit model to validator
            {"name": names.validate, "length": cycle_cfg.validate_period},  # validator validate models from miners
            {"name": names.merge, "length": cycle_cfg.merge_period},  # validator merge models
            {
                "name": names.validator_commit_1,
                "length": cycle_cfg.commit_period,
            },  # validator commit signed_model_hash
            {"name": names.validator_commit_2, "length": cycle_cfg.commit_period},  # validator commit model_hash
        ]
        return phases

    def get_phase(self, block: int = None) -> PhaseResponse:
        if block is None:
            block = self.subtensor.block

        if block < 0:
            raise RuntimeError(f"Invalida block input block = {block}")

        cycle_index = block // self.cycle_length
        cycle_block_index = block % self.cycle_length  # 0..self.cycle_length-1

        # Walk through phases to find which one cycle_block_index is in
        current_start = 0
        for idx, phase in enumerate(self.phases):
            phase_end = current_start + phase["length"]  # exclusive
            if current_start <= cycle_block_index < phase_end:
                blocks_into_phase = cycle_block_index - current_start
                blocks_remaining = phase_end - cycle_block_index - 1

                return PhaseResponse(
                    block=block,
                    cycle_length=self.cycle_length,
                    cycle_index=cycle_index,
                    cycle_block_index=cycle_block_index,
                    phase_name=phase["name"],
                    phase_index=idx,
                    phase_start_block=current_start + cycle_index * self.cycle_length,
                    phase_end_block=phase_end - 1 + cycle_index * self.cycle_length,
                    blocks_into_phase=blocks_into_phase,
                    blocks_remaining_in_phase=blocks_remaining,
                )
            current_start = phase_end

        # Should never happen if PHASES and self.cycle_length are consistent
        raise RuntimeError("Failed to determine phase")

    def blocks_until_next_phase(self) -> dict[str, tuple[int, int, int]]:
        """
        Returns a mapping:
            { phase_name: (start_block, end_block, blocks_until) }

        Each tuple corresponds to the next occurrence of that phase (in this
        cycle if upcoming, otherwise the next cycle).
        """
        block = self.subtensor.block
        cycle_len = self.cycle_length
        cycle_block_index = block % cycle_len  # 0..cycle_len-1
        cycle_start_block = block - cycle_block_index

        result: dict[str, tuple[int, int, int]] = {}

        start = 0
        for phase in self.phases:
            phase_start = start  # phase start within the cycle [0..cycle_len)
            phase_end = start + phase["length"] - 1

            if phase_start >= cycle_block_index:
                # Phase still ahead in *this* cycle
                start_block = cycle_start_block + phase_start
            else:
                # Phase has already passed in this cycle -> wait for next cycle
                start_block = cycle_start_block + cycle_len + phase_start

            end_block = start_block + phase["length"] - 1
            blocks_until = start_block - block

            result[phase["name"]] = (start_block, end_block, blocks_until)
            start += phase["length"]

        return result

    def previous_phase_block_ranges(self) -> dict[str, tuple[int, int]]:
        """
        Returns a mapping:
            { phase_name: (start_block, end_block) }

        The range corresponds to the most recent completed
        occurrence of that phase.
        """
        next_phase_ranges = self.blocks_until_next_phase()
        previous_ranges: dict[str, tuple[int, int]] = {}
        for phase_name, (start_block, end_block, _blocks_until) in next_phase_ranges.items():
            previous_ranges[phase_name] = (
                start_block - self.cycle_length,
                end_block - self.cycle_length,
            )
        return previous_ranges