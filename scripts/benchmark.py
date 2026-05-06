"""
Compute BLEU, ROUGE, BERT scores and exact-match on JSON files written by generate.py.

Single-process, single-GPU.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List

import evaluate
from transformers import BertTokenizer, RobertaTokenizer

from .utils import str2bool


argParser = argparse.ArgumentParser()

argParser.add_argument("--read_generation_dir", type=str)
argParser.add_argument("--read_file_identifier", type=str, help="Postfix identifier or timestamp to filter files.")

argParser.add_argument("--evaluate_exact_match", type=str2bool)
argParser.add_argument("--evaluate_bleu", type=str2bool)
argParser.add_argument("--evaluate_rouge", type=str2bool)
argParser.add_argument("--evaluate_bert_score", type=str2bool)
argParser.add_argument("--verbose", type=str2bool)


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute exact match ratio allowing for case and punctuation differences."""
    def normalize(text: str) -> str:
        """Normalize text by lowercasing and removing punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w]', '', text)
        return text

    exact_match = 0
    for pred, ref in zip(predictions, references):
        if normalize(pred) == normalize(ref):
            exact_match += 1
    return exact_match / len(predictions)


def compute_bleu2(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=references, max_order=2)


def compute_bleu4(predictions: List[str], references: List[str]) -> Dict[str, Any]: 
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=references)


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, Any]: 
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def compute_bert_score(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, Any]]: 
    """Compute BERT score on roberta-large and biobert-large respectively."""
    results: Dict[str, Dict[str, Any]] = {}

    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-large")
    retokenized_predictions = tokenizer(
        predictions, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    truncated_predictions = tokenizer.batch_decode(retokenized_predictions, skip_special_tokens=True)
    retokenized_labels = tokenizer(
        references, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    truncated_labels = tokenizer.batch_decode(retokenized_labels, skip_special_tokens=True)

    bert = evaluate.load("bertscore")
    roberta_results = bert.compute(predictions=truncated_predictions, references=truncated_labels, lang="en")
    results["roberta-large"] = {
        "precision": sum(roberta_results["precision"]) / len(roberta_results["precision"]), 
        "recall": sum(roberta_results["recall"]) / len(roberta_results["recall"]), 
        "f1": sum(roberta_results["f1"]) / len(roberta_results["f1"])
    }

    # truncate to fit biobert's max_position_embeddings=512
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1")
    retokenized_predictions = tokenizer(
        predictions, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    truncated_predictions = tokenizer.batch_decode(retokenized_predictions, skip_special_tokens=True)
    retokenized_labels = tokenizer(
        references, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    truncated_labels = tokenizer.batch_decode(retokenized_labels, skip_special_tokens=True)

    biobert_results = bert.compute(
        predictions=truncated_predictions,
        references=truncated_labels,
        model_type="dmis-lab/biobert-large-cased-v1.1",
        num_layers=24,
    )
    results["biobert-large"] = {
        "precision": sum(biobert_results["precision"]) / len(biobert_results["precision"]), 
        "recall": sum(biobert_results["recall"]) / len(biobert_results["recall"]), 
        "f1": sum(biobert_results["f1"]) / len(biobert_results["f1"])
    }

    return results


def compute_metrics(predictions: List[str], references: List[str], args: Dict[str, Any]) -> Dict[str, Any]:
    """Compute BLEU, ROUGE, BERT scores and exact match ratio on given texts."""
    gathered_results: Dict[str, Dict[str, Any]] = {}

    if args["evaluate_exact_match"]:
        exact_match = compute_exact_match(predictions=predictions, references=references)
        gathered_results["exact_match"] = exact_match
        if args["verbose"]:
            print(f"EXACT match ratio: {exact_match}")

    if args["evaluate_bleu"]:
        bleu_results = compute_bleu2(predictions=predictions, references=references)
        gathered_results["bleu2"] = bleu_results
        if args["verbose"]:
            print(f"BLEU-2 score: {bleu_results}")
        bleu_results = compute_bleu4(predictions=predictions, references=references)
        gathered_results["bleu4"] = bleu_results
        if args["verbose"]:
            print(f"BLEU-4 score: {bleu_results}")

    if args["evaluate_rouge"]:
        rouge_results = compute_rouge(predictions=predictions, references=references)
        gathered_results["rouge"] = rouge_results
        if args["verbose"]:
            print(f"ROUGE score: {rouge_results}")
    
    if args["evaluate_bert_score"]:
        bert_results = compute_bert_score(predictions=predictions, references=references)
        gathered_results["bert"] = bert_results
        if args["verbose"]:
            for model_name, model_results in bert_results.items(): 
                print(f"BERT score with {model_name}: {model_results}")

    return gathered_results


def benchmark(args: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Gather predictions/labels from generate.py outputs and compute BLEU/ROUGE/BERT/exact-match."""
    read_generation_paths = []
    for file_name in os.listdir(args["read_generation_dir"]):
        full_path = os.path.join(args["read_generation_dir"], file_name)
        if (os.path.isfile(full_path)
                and file_name.endswith(".json")
                and args["read_file_identifier"] in full_path):
            read_generation_paths.append(full_path)

    gathered_predictions = []
    gathered_labels = []
    for read_path in read_generation_paths:
        with open(read_path, "r") as file: 
            results = json.load(file)
            local_predictions = [results[name]["pred"] for name in results.keys()]
            local_labels = [results[name]["true"] for name in results.keys()]
            gathered_predictions.extend(local_predictions)
            gathered_labels.extend(local_labels)
        print(f"Reading {read_path}")

    return compute_metrics(predictions=gathered_predictions, references=gathered_labels, args=args)


if __name__ == "__main__": 
    parsed_args = argParser.parse_args()

    print("####################")
    for key, value in parsed_args.__dict__.items(): 
        print(f"{key}: {value}")
    print("####################")

    benchmark(parsed_args.__dict__)
