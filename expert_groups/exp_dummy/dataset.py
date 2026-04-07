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
        Default data formatting function.

        This function demonstrates how to transform a raw dataset row
        into tokenized tensors suitable for training. Customers may
        freely modify or replace this function to implement custom
        formatting logic.

        Expected Input
        --------------
        example : Dict[str, Any]
            A single raw dataset row containing a list of chat-style messages
            under `example["messages"]`.

        tokenizer : PreTrainedTokenizerBase
            HuggingFace tokenizer used to build and tokenize the text sequence.

        sequence_length : int
            Maximum sequence length for tokenization and padding.

        Expected Output
        ---------------
        Dict[str, Any]
            Dictionary containing tokenized tensors:

                {
                    "input_ids": Tensor[1, sequence_length],
                    "attention_mask": Tensor[1, sequence_length]
                }

            Additional fields may be added if required by your model.

        Notes for Customization
        -----------------------
        - You may change how messages are converted to text.
        - You can modify tokenization parameters (padding, truncation, etc.).
        - You can inject additional metadata into the output dictionary.
        - You can apply your own chat template logic.

        Returns
        -------
        Dict[str, Any]
            Tokenized output ready to be consumed by a DataLoader.
        """
        # 1) Convert dataset row → chat messages
        messages = example["messages"]

        # 2) Convert messages → raw text using model's chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 3) Tokenize text → tensors
        toks = tokenizer(
            text,
            truncation=True,
            max_length=sequence_length,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="pt",
        )

        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }
