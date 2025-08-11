# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0]
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.


import argparse
import json
import os
from pathlib import Path

import cv2
import torch
import torch.distributed as dist
from accelerate.logging import get_logger

from fastvideo.utils.dataset_utils import get_all_data

logger = get_logger(__name__)
import re

from diffusers import FluxPipeline
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def contains_chinese(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text))


class T5dataset(Dataset):
    def __init__(
        self,
        json_path,
    ):
        self.json_path = json_path
        self.train_dataset = get_all_data(json_path)

    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        caption = self.train_dataset[idx]
        filename = str(idx)
        # length = self.train_dataset[idx]["length"]
        latents = []

        return dict(caption=caption, latents=latents, filename=filename)

    def __len__(self):
        return len(self.train_dataset)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "text_ids"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "pooled_prompt_embeds"), exist_ok=True)

    latents_txt_path = args.prompt_path
    train_dataset = T5dataset(latents_txt_path)
    sampler = DistributedSampler(train_dataset, rank=local_rank, num_replicas=world_size, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    pipe = FluxPipeline.from_pretrained("./data/flux", torch_dtype=torch.bfloat16).to(device)
    # pipe = FluxPipeline.from_pretrained("./data/flux").to(device)

    json_data = []
    for i, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        with torch.inference_mode():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=data["prompt"], prompt_2=data["prompt"]
            )
            prompt_embed_path = os.path.join(args.output_dir, "prompt_embed", str(i) + ".pt")
            pooled_prompt_embeds_path = os.path.join(args.output_dir, "pooled_prompt_embeds", str(i) + ".pt")

            text_ids_path = os.path.join(args.output_dir, "text_ids", str(i) + ".pt")
            # save latent
            torch.save(prompt_embeds, prompt_embed_path)
            torch.save(pooled_prompt_embeds, pooled_prompt_embeds_path)
            torch.save(text_ids, text_ids_path)
            item = {}
            item["prompt_embed_path"] = str(i) + ".pt"
            item["text_ids"] = str(i) + ".pt"
            item["pooled_prompt_embeds_path"] = str(i) + ".pt"
            item["caption"] = data["caption"]
            json_data.append(item)
    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        # os.remove(latents_json_path)
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "prompt.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    # text encoder & vae & diffusion model
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--text_encoder_name", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--vae_debug", action="store_true")
    parser.add_argument("--prompt_path", type=str, default="./empty.txt")
    args = parser.parse_args()
    main(args)

