from typing import Any

from transformers import PreTrainedTokenizerBase

from connito.shared.dataloader import DefaultStreamingTorchDataset


# -------------------------------------------------------------
# Customer Extension Point: Customize how your dataset is loaded
# make sure this class was pointed to in the config through config.task.exp.data.dataset_class
# -------------------------------------------------------------
class StreamingTorchDataset(DefaultStreamingTorchDataset):
    @staticmethod
    def tokenize_and_format(
        example: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
    ) -> dict[str, Any]:
        """
        Processes raw text for Continuous Pre-Training (CPT).
        Bypasses the chat template since C4 and Nemotron-CC-Math 
        are raw text corpora, not conversational data.
        """
        # 1) Safely extract the raw text we aligned in dataloader.py
        text = str(example.get("text", ""))

        # 2) Tokenize text directly (no chat template)
        toks = tokenizer(
            text,
            truncation=True,
            max_length=sequence_length,
            padding="max_length",
            add_special_tokens=True, # Important: Ensures BOS/EOS tokens are added
        )

        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }