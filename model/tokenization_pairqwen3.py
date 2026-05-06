"""
Fast tokenizer of Qwen3 adapted for protein-protein interaction pair model.
Special tokens: <|boi|> (begin of interaction), <|eoi|> (end of interaction), <|roi|> (representation of interaction).

The vocabulary size of Qwen3-8B is 151,936 and leaves space for additional special tokens. Thus, the embedding layer of
the model can accommodate the additional tokens without resizing.

Example of usage:
>>> tokenizer = PairQwen3Tokenizer.from_pretrained("Qwen/Qwen3-8B")
>>> chat = [{"role": "user", "content": "describe the interaction between <|boi|><|roi|><|roi|><|roi|><|eoi|>."}]
>>> tokenizer.padding_side = "left"
>>> tokenized_prompts = tokenizer.apply_chat_template(
        [chat],
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=True,
        padding="longest",
        return_tensors="pt",
        return_dict=True
    )
"""

from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer


class PairQwen3Tokenizer(Qwen2Tokenizer):
    def __init__(
            self,
            *args,
            boi_token: str = "<|boi|>",
            eoi_token: str = "<|eoi|>",
            roi_token: str = "<|roi|>",
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._boi_token = boi_token
        self._eoi_token = eoi_token
        self._roi_token = roi_token

        self.add_special_tokens({
            "additional_special_tokens": [self._boi_token, self._eoi_token, self._roi_token]
        })

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        # from_pretrained method will first call __init__ method of this customized class
        # then immediately overwrite tokenizer states, including the special tokens, from designated pretrained files
        # thus we need to add the special tokens again to ensure they are included in the vocabulary
        tokenizer.add_special_tokens({
            "additional_special_tokens": [tokenizer.boi_token, tokenizer.eoi_token, tokenizer.roi_token]
        })
        return tokenizer

    def save_pretrained(self, *args, **kwargs):
        raise NotImplementedError(
            "Saving is not supported for PairQwen3Tokenizer; load a fresh tokenizer from Qwen/Qwen3-8B each time."
        )

    @property
    def boi_token(self):
        return self._boi_token

    @property
    def eoi_token(self):
        return self._eoi_token

    @property
    def roi_token(self):
        return self._roi_token

    @property
    def boi_token_id(self):
        return self.convert_tokens_to_ids(self.boi_token)

    @property
    def eoi_token_id(self):
        return self.convert_tokens_to_ids(self.eoi_token)

    @property
    def roi_token_id(self):
        return self.convert_tokens_to_ids(self.roi_token)
