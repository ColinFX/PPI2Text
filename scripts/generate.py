"""
Distributed generation for PairEsm3Qwen3ForCausalLM.

Uses HF Accelerate for DDP, loads the full model with merged LoRA weights, runs autoregressive generation per pair,
and writes per-rank JSON: {pair_id: {"true": label, "pred": prediction}}. Multi-GPU, multi-node.
"""

import argparse
from datetime import datetime
import json
import os
from typing import Any, Dict, List

from accelerate import Accelerator
from accelerate.utils import set_seed
from peft.peft_model import PeftModel
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from dataworker import PairSFTDataset, PairSFTCollator
from model import (
    OpenESM3,
    PairEsm3Qwen3Config,
    PairEsm3Qwen3ForCausalLM,
    PairQwen3ForCausalLM,
    get_openesm3_model_tokenizers,
    PairQwen3Tokenizer,
)
from .utils import str2bool, str2dtype


argParser = argparse.ArgumentParser()

argParser.add_argument("--generate_data_path", type=str, help="Path to inference data (Parquet or JSONL).")
argParser.add_argument("--generate_list_path", type=str, default="", help="Optional pair_id allow-list, one per line.")
argParser.add_argument("--proteins_jsonl_path", type=str, default="", help="Optional protein sequence jsonl file.")
argParser.add_argument("--read_pt_dir", type=str)
argParser.add_argument("--read_emb_dir", type=str, default="")
argParser.add_argument("--esm_path", type=str, default="")
argParser.add_argument("--qwen_path", type=str)
argParser.add_argument("--save_generation_dir", type=str)
argParser.add_argument("--save_generation_postfix_identifier", type=str, default=None)

argParser.add_argument("--load_adapter_checkpoint_dir", type=str, default="")

argParser.add_argument("--compress_d", type=int, default=1024)
argParser.add_argument("--n_cross_layers", type=int, default=2)
argParser.add_argument("--n_cross_heads", type=int, default=8)
argParser.add_argument("--d_pair", type=int, default=1024)
argParser.add_argument("--d_pair_constructor_mid", type=int, default=2048)
argParser.add_argument("--d_pair_map_mid", type=int, default=1536)
argParser.add_argument("--pair_map_target_h", type=int, default=32)
argParser.add_argument("--pair_map_target_w", type=int, default=32)
argParser.add_argument("--mrope_section", type=str, default="24,20,20")

argParser.add_argument("--torch_dtype", type=str2dtype)
argParser.add_argument("--batch_size_per_device", type=int)
argParser.add_argument("--random_seed", type=int, default=42)
argParser.add_argument("--max_new_tokens", type=int, default=512)
argParser.add_argument("--num_beams", type=int, default=1)
argParser.add_argument("--length_penalty", type=float, default=1.0)
argParser.add_argument("--temperature", type=float, default=1.0)
argParser.add_argument("--do_sample", type=str2bool, default=False)
argParser.add_argument("--top_p", type=float, default=1.0)
argParser.add_argument("--top_k", type=int, default=50)
argParser.add_argument("--max_seq_len", type=int, default=2040)


def load_model(args):
    """Load PairEsm3Qwen3ForCausalLM with merged LoRA weights for generation."""
    use_embeddings = bool(args.get("read_emb_dir"))

    if use_embeddings:
        esm_encoder = None
        print("Using pre-computed ESM3 embeddings — skipping ESM3 model loading")
    else:
        esm_encoder = OpenESM3.from_pretrained(local_dir=args["esm_path"], device=torch.device("cpu"))
        if next(esm_encoder.parameters()).dtype != args["torch_dtype"]:
            esm_encoder = esm_encoder.to(args["torch_dtype"])

    qwen_decoder = PairQwen3ForCausalLM.from_pretrained(
        args["qwen_path"],
        sliding_window=None,
        dtype=args["torch_dtype"],
        device_map="cpu",
    )

    mrope_section = [int(x) for x in args["mrope_section"].split(",")]

    config = PairEsm3Qwen3Config(
        esm3_d_model=esm_encoder.d_model if esm_encoder is not None else 1536,
        esm3_n_heads=esm_encoder.n_heads if esm_encoder is not None else 24,
        esm3_v_heads=esm_encoder.v_heads if esm_encoder is not None else 256,
        esm3_n_layers=esm_encoder.n_layers if esm_encoder is not None else 48,
        esm3_tokenizers=esm_encoder.tokenizers if esm_encoder is not None else None,
        compress_d=args["compress_d"],
        n_cross_layers=args["n_cross_layers"],
        n_cross_heads=args["n_cross_heads"],
        d_pair=args["d_pair"],
        d_pair_constructor_mid=args["d_pair_constructor_mid"],
        d_pair_map_mid=args["d_pair_map_mid"],
        pair_map_target_h=args["pair_map_target_h"],
        pair_map_target_w=args["pair_map_target_w"],
        mrope_section=mrope_section,
        qwen_config=qwen_decoder.config,
    )

    model = PairEsm3Qwen3ForCausalLM(
        config=config,
        esm_encoder=esm_encoder,
        qwen_decoder=qwen_decoder,
    )

    # load and merge LoRA adapter
    if args["load_adapter_checkpoint_dir"]:
        print(f"Merging LoRA: {args['load_adapter_checkpoint_dir']}")
        lora_model = PeftModel.from_pretrained(model, args["load_adapter_checkpoint_dir"], is_trainable=True)
        lora_model = lora_model.to(args["torch_dtype"])
        model = lora_model.merge_and_unload()
    else:
        print("WARNING: No adapter checkpoint loaded for generation.")

    return model


@torch.no_grad()
def iterative_generation_loop(rank, model, data_batch, args):
    protein_a = {k: v.to(rank) for k, v in data_batch["protein_a"].items()}
    protein_b = {k: v.to(rank) for k, v in data_batch["protein_b"].items()}

    gen_model = model.module if hasattr(model, "module") else model
    with torch.amp.autocast(device_type="cuda", dtype=args["torch_dtype"]):
        return gen_model.generate(
            inputs=data_batch["input_ids"].to(rank),
            attention_mask=data_batch["attention_mask"].to(rank),
            protein_a=protein_a,
            protein_b=protein_b,
            compressed_len_a=data_batch["compressed_len_a"].to(rank),
            compressed_len_b=data_batch["compressed_len_b"].to(rank),
            max_new_tokens=args["max_new_tokens"],
            eos_token_id=151645,
            pad_token_id=151645,
            return_dict_in_generate=False,
            num_beams=args["num_beams"],
            length_penalty=args["length_penalty"],
            temperature=args["temperature"],
            do_sample=args["do_sample"],
            top_p=args["top_p"],
            top_k=args["top_k"],
        )


@torch.no_grad()
def inference_epoch(rank, model, dataloader, text_tokenizer, args):
    model.eval()
    local_pair_ids: List[str] = []
    local_predictions: List[str] = []
    local_labels: List[str] = []

    t = tqdm(iter(dataloader))
    for data_batch in t:
        output = iterative_generation_loop(rank, model, data_batch, args)
        local_pair_ids.extend(data_batch["pair_ids"])
        predicted_texts = text_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)
        local_predictions.extend(predicted_texts)
        label_texts = text_tokenizer.batch_decode(data_batch["labels"], skip_special_tokens=True)
        local_labels.extend(label_texts)
        t.set_postfix({"maxlen": output.shape[1], "rank": rank})

    json_path = os.path.join(
        args["save_generation_dir"],
        f"generation_{args['save_generation_postfix_identifier']}_rank{rank}.json"
    )
    with open(json_path, "w") as f:
        json_dict = {
            pid: {"true": label, "pred": pred}
            for pid, label, pred in zip(local_pair_ids, local_labels, local_predictions)
        }
        json.dump(json_dict, f, indent=4)
    print(f"Saved {json_path}")


def inference_distributed(args):
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print("####################")
        for k, v in args.items():
            print(f"{k}: {v}")
        print("####################")

    set_seed(args["random_seed"])

    use_embeddings = bool(args.get("read_emb_dir"))
    protein_tokenizers = get_openesm3_model_tokenizers(local_dir=args["esm_path"]) if not use_embeddings else None
    qwen_tokenizer = PairQwen3Tokenizer.from_pretrained(args["qwen_path"])

    gen_dataset = PairSFTDataset(
        read_data_path=args["generate_data_path"],
        read_pt_dir=args["read_pt_dir"],
        text_tokenizer=qwen_tokenizer,
        protein_tokenizers=protein_tokenizers,
        max_seq_len=args["max_seq_len"],
        read_emb_dir=args.get("read_emb_dir") or None,
        read_list_path=args.get("generate_list_path") or None,
        proteins_jsonl_path=args.get("proteins_jsonl_path") or None,
    )
    data_collator = PairSFTCollator(
        protein_tokenizers=protein_tokenizers,
        text_tokenizer=qwen_tokenizer,
        mode="inference",
        pair_map_n_tokens=args["pair_map_target_h"] * args["pair_map_target_w"],
    )
    gen_dataloader = DataLoader(
        gen_dataset, batch_size=args["batch_size_per_device"],
        collate_fn=data_collator, num_workers=4, pin_memory=True, shuffle=False, drop_last=True
    )

    model = load_model(args)
    model, gen_dataloader = accelerator.prepare(model, gen_dataloader)

    inference_epoch(accelerator.process_index, model, gen_dataloader, qwen_tokenizer, args)

    if accelerator.distributed_type != "NO":
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parsed_args = argParser.parse_args()
    parsed_args.world_size = torch.cuda.device_count()

    torch.manual_seed(parsed_args.random_seed)
    torch.cuda.manual_seed(parsed_args.random_seed)

    os.makedirs(parsed_args.save_generation_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    if parsed_args.save_generation_postfix_identifier:
        parsed_args.save_generation_postfix_identifier = (
            f"{timestamp}_[{parsed_args.save_generation_postfix_identifier}]"
        )
    else:
        parsed_args.save_generation_postfix_identifier = timestamp

    inference_distributed(parsed_args.__dict__)
