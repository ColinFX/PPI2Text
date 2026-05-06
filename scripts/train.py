"""
DistributedDataParallel training script implemented from scratch. 

The script currently supports gradient accumulation, inter-epoch validation, save/load pretrained checkpoints.  
The script also support mixed precision, as the base model will be loaded in the predefined torch_dtype but the lora 
adapter, the optimizer and the gradients will be in full precision for stable training.
The script currently does not support gradient checkpointing due to conflicts with LoRA. 

* The script is designed for multi-node multi-GPU parallelism.
* On the cluster, print(...) will go to stdout and tqdm(...) will go to stderr.
"""

import argparse
from datetime import datetime
import os
from time import sleep

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from peft import get_peft_model, LoraConfig
from peft.peft_model import PeftModel
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

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

argParser.add_argument("--train_data_path", type=str, help="Path to training data (Parquet or JSONL).")
argParser.add_argument("--val_data_path", type=str, help="Path to validation data (Parquet or JSONL).")
argParser.add_argument("--train_list_path", type=str, default="", help="Optional pair_id list file for train split.")
argParser.add_argument("--val_list_path", type=str, default="", help="Optional pair_id list file for val split.")
argParser.add_argument("--proteins_jsonl_path", type=str, default="", help="Optional protein sequence jsonl file.")
argParser.add_argument("--read_pt_dir", type=str, default="", help="Directory containing pre-encoded proteins from ESM3.")
argParser.add_argument("--read_emb_dir", type=str, default="", help="Directory containing pre-embedded proteins from ESM3.")
argParser.add_argument("--esm_path", type=str, default="", help="Path to ESM3 checkpoint or local dir.")
argParser.add_argument("--qwen_path", type=str, help="Path to Qwen checkpoint or local dir.")
argParser.add_argument("--save_checkpoint_dir", type=str, help="Directory to save checkpoints.")

argParser.add_argument("--load_adapter_checkpoint_dir", type=str, default="")
argParser.add_argument("--rewrap_lora", type=str2bool, default=False)
argParser.add_argument("--load_optimizer_scheduler_checkpoint_path", type=str, default="")
argParser.add_argument("--restart_scheduler", type=str2bool, default=False)

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
argParser.add_argument("--num_epochs", type=int)
argParser.add_argument("--save_every_steps", type=int, default=None)
argParser.add_argument("--save_every_epochs", type=int, default=1)
argParser.add_argument("--gradient_accumulation_steps", type=int, default=1)
argParser.add_argument("--learning_rate_lora", type=float)
argParser.add_argument("--learning_rate_from_scratch", type=float)
argParser.add_argument("--gradient_clipping", type=float, default=None)
argParser.add_argument("--scheduler_warmup_steps", type=int)
argParser.add_argument("--scheduler_gamma", type=float)
argParser.add_argument("--random_seed", type=int, default=42)
argParser.add_argument("--lora_rank", type=int, default=16)
argParser.add_argument("--max_seq_len", type=int, default=2040)
argParser.add_argument("--lora_dropout", type=float, default=0.1)


def load_model(args):
    """Initialize PairEsm3Qwen3ForCausalLM and wrap with LoRA."""
    use_embeddings = bool(args.get("read_emb_dir"))

    if use_embeddings:
        esm_encoder = None
        print("Using pre-computed ESM3 embeddings, skipping ESM3 model loading")
    else:
        esm_encoder = OpenESM3.from_pretrained(local_dir=args["esm_path"], device=torch.device("cpu"))
        if next(esm_encoder.parameters()).dtype != args["torch_dtype"]:
            esm_encoder = esm_encoder.to(args["torch_dtype"])

    qwen_decoder = PairQwen3ForCausalLM.from_pretrained(
        args["qwen_path"],
        sliding_window=None,
        torch_dtype=args["torch_dtype"],
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

    # load existing LoRA adapter checkpoint if resuming
    if args["load_adapter_checkpoint_dir"]:
        print(f"Loading LoRA adapter: {args['load_adapter_checkpoint_dir']}")
        model = PeftModel.from_pretrained(model, args["load_adapter_checkpoint_dir"], is_trainable=True)
        if args["rewrap_lora"]:
            print("Merging and re-wrapping LoRA")
            model = model.merge_and_unload()

    if not args["load_adapter_checkpoint_dir"] or args["rewrap_lora"]:
        print("Initializing new LoRA adapter")
        modules_to_save = ["compressor"] + [
            f"cross_attn_layers.{i}" for i in range(args["n_cross_layers"])
        ] + [
            "protein_projector",
            "pair_map_constructor",
            "pair_map_to_tokens",
            "boi_embed",
            "eoi_embed",
        ]

        lora_config = LoraConfig(
            r=args["lora_rank"],
            lora_alpha=args["lora_rank"] * 2,
            lora_dropout=args["lora_dropout"],
            bias="none",
            init_lora_weights=True,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ],
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)

        # upcast modules_to_save to float32 for stable training
        for name, param in model.named_parameters():
            if "modules_to_save" in name:
                param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()
    return model


def teacher_forcing_forward_pass(local_rank, model, data_batch, args):
    protein_a = {k: v.to(local_rank) for k, v in data_batch["protein_a"].items()}
    protein_b = {k: v.to(local_rank) for k, v in data_batch["protein_b"].items()}

    with torch.amp.autocast(device_type="cuda", dtype=args["torch_dtype"]):
        return model(
            input_ids=data_batch["input_ids"].to(local_rank),
            attention_mask=data_batch["attention_mask"].to(local_rank),
            labels=data_batch["labels"].to(local_rank),
            protein_a=protein_a,
            protein_b=protein_b,
            compressed_len_a=data_batch["compressed_len_a"].to(local_rank),
            compressed_len_b=data_batch["compressed_len_b"].to(local_rank),
            use_cache=False,
        ).loss


def setup():
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"[SETUP] rank:{global_rank}({local_rank})")
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '9901')
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    return local_rank, global_rank, world_size


def cleanup():
    dist.destroy_process_group()


def save_checkpoint(postfix, model, optimizer, scheduler, args):
    adapter_dir = os.path.join(args["save_checkpoint_dir"], f"adapter_checkpoint_{postfix}")
    model.module.save_pretrained(adapter_dir)
    print(f"Saved {adapter_dir}")

    opt_path = os.path.join(args["save_checkpoint_dir"], f"optimizer_scheduler_checkpoint_{postfix}.pt")
    torch.save({"optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()}, opt_path)
    print(f"Saved {opt_path}")


def train_epoch(global_rank, local_rank, current_epoch, model, dataloader, optimizer, scheduler, args):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)
    ddp_gradnorm = torch.zeros(2).to(local_rank)
    optimizer.zero_grad(set_to_none=True)

    t = tqdm(iter(dataloader), desc=f"Train epoch {current_epoch}/{args['num_epochs']}")
    for batch_idx, data_batch in enumerate(t):
        loss = teacher_forcing_forward_pass(local_rank, model, data_batch, args)
        loss = loss.to(torch.float32) / args["gradient_accumulation_steps"]

        t.set_postfix({
            "loss": loss.item() * args["gradient_accumulation_steps"],
            "lr": optimizer.param_groups[0]['lr'],
            "rank": f"{global_rank}({local_rank})"
        })
        ddp_loss[0] += loss.item() * args["gradient_accumulation_steps"]
        ddp_loss[1] += 1

        if (batch_idx + 1) % args["gradient_accumulation_steps"] != 0:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
            current_step = (batch_idx + 1) // args["gradient_accumulation_steps"]

            gradnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf") if args["gradient_clipping"] is None else args["gradient_clipping"]
            )
            if not torch.isfinite(gradnorm):
                raise ValueError("Non-finite gradient norm detected.")

            ddp_gradnorm[0] += gradnorm
            ddp_gradnorm[1] += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if args["save_every_steps"] is not None and current_step % args["save_every_steps"] == 0:
                dist.barrier()
                if global_rank == 0:
                    save_checkpoint(f"{current_epoch}_{current_step}", model, optimizer, scheduler, args)
                dist.barrier()

    if (
        args["save_every_epochs"] is not None
        and current_epoch % args["save_every_epochs"] == 0
    ):
        dist.barrier()
        if global_rank == 0:
            save_checkpoint(str(current_epoch), model, optimizer, scheduler, args)
        dist.barrier()

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if global_rank == 0:
        avg_loss = ddp_loss[0] / max(ddp_loss[1], 1)
        avg_gn = ddp_gradnorm[0] / max(ddp_gradnorm[1], 1)
        print(f"[epoch={current_epoch}, train_loss={avg_loss:.6f}, gradnorm={avg_gn:.4f}]")
        if avg_loss != avg_loss:
            raise ValueError("NaN loss detected.")


def val_epoch(global_rank, local_rank, current_epoch, model, dataloader, args):
    model.eval()
    ddp_loss = torch.zeros(2).to(local_rank)

    t = tqdm(iter(dataloader), desc=f"Val epoch {current_epoch}/{args['num_epochs']}")
    for data_batch in t:
        with torch.no_grad():
            loss = teacher_forcing_forward_pass(local_rank, model, data_batch, args)
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if global_rank == 0:
        print(f"[epoch={current_epoch}, val_loss={ddp_loss[0] / max(ddp_loss[1], 1):.6f}]")


def train_on_device(args):
    local_rank, global_rank, world_size = setup()

    use_embeddings = bool(args.get("read_emb_dir"))
    protein_tokenizers = get_openesm3_model_tokenizers(local_dir=args["esm_path"]) if not use_embeddings else None
    qwen_tokenizer = PairQwen3Tokenizer.from_pretrained(args["qwen_path"])

    train_dataset = PairSFTDataset(
        read_data_path=args["train_data_path"],
        read_pt_dir=args["read_pt_dir"],
        text_tokenizer=qwen_tokenizer,
        protein_tokenizers=protein_tokenizers,
        max_seq_len=args["max_seq_len"],
        read_emb_dir=args.get("read_emb_dir") or None,
        read_list_path=args.get("train_list_path") or None,
        proteins_jsonl_path=args.get("proteins_jsonl_path") or None,
    )
    data_collator = PairSFTCollator(
        protein_tokenizers=protein_tokenizers,
        text_tokenizer=qwen_tokenizer,
        mode="train_val",
        pair_map_n_tokens=args["pair_map_target_h"] * args["pair_map_target_w"],
    )
    train_sampler = DistributedSampler(
        train_dataset, 
        rank=global_rank, 
        num_replicas=world_size, 
        shuffle=True, 
        seed=args["random_seed"]
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args["batch_size_per_device"], 
        collate_fn=data_collator,
        sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    print(f"Train dataset loaded on rank:{global_rank}({local_rank})")

    val_dataset = PairSFTDataset(
        read_data_path=args["val_data_path"],
        read_pt_dir=args["read_pt_dir"],
        text_tokenizer=qwen_tokenizer,
        protein_tokenizers=protein_tokenizers,
        max_seq_len=args["max_seq_len"],
        read_emb_dir=args.get("read_emb_dir") or None,
        read_list_path=args.get("val_list_path") or None,
        proteins_jsonl_path=args.get("proteins_jsonl_path") or None,
    )
    val_sampler = DistributedSampler(
        val_dataset, 
        rank=global_rank, 
        num_replicas=world_size, 
        shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args["batch_size_per_device"], 
        collate_fn=data_collator,
        sampler=val_sampler, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )

    torch.cuda.set_device(local_rank)
    model = load_model(args).to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    # separate LR for LoRA vs from-scratch parameters
    lora_params, scratch_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            scratch_params.append(param)
    optimizer = Adam([
        {"params": lora_params, "lr": args["learning_rate_lora"]},
        {"params": scratch_params, "lr": args["learning_rate_from_scratch"]},
    ])

    def lr_lambda(step):
        if step < args["scheduler_warmup_steps"]:
            return (step + 1) / args["scheduler_warmup_steps"]
        return args["scheduler_gamma"] ** (step + 1 - args["scheduler_warmup_steps"])

    scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda, lr_lambda])

    if args["load_optimizer_scheduler_checkpoint_path"]:
        ckpt = torch.load(args["load_optimizer_scheduler_checkpoint_path"], weights_only=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if not args["restart_scheduler"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    for epoch in range(1, args["num_epochs"] + 1):
        train_sampler.set_epoch(epoch)
        train_epoch(global_rank, local_rank, epoch, model, train_loader, optimizer, scheduler, args)
        dist.barrier()
        val_epoch(global_rank, local_rank, epoch, model, val_loader, args)
        dist.barrier()

    cleanup()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parsed_args = argParser.parse_args()

    if "RANK" not in os.environ:
        raise ValueError("RANK not set. Use torchrun or srun.")

    torch.manual_seed(parsed_args.random_seed)
    torch.cuda.manual_seed(parsed_args.random_seed)

    if int(os.environ["RANK"]) == 0:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        parsed_args.save_checkpoint_dir = os.path.join(parsed_args.save_checkpoint_dir, f"run_{timestamp}")
        os.makedirs(parsed_args.save_checkpoint_dir, exist_ok=True)
        print("####################")
        for k, v in parsed_args.__dict__.items():
            print(f"{k}: {v}")
        print("####################")
    else:
        sleep(3)

    dist.init_process_group(backend="gloo")
    dist.barrier()
    dist.destroy_process_group()

    train_on_device(parsed_args.__dict__)
