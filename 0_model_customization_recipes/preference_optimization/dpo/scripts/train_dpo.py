from transformers import (
    AutoModelForCausalLM,
    set_seed,
)
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
import logging
import os
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import TrlParser, ModelConfig, get_peft_config
from datasets import load_dataset
from trl import (
    DPOTrainer,
    DPOConfig,
    TrlParser,
    get_peft_config,
    ModelConfig,
)

from datasets import load_dataset
from huggingface_hub import snapshot_download

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin


from typing import Any, Dict, List, Optional, Union

#This can be set if running into CUDA OOM issues
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = "/opt/ml/model/input"
    model_download_location: str = "/opt/ml/model/input"
    dataset_local_location: str = "/opt/ml/input/data/training_dataset/"
    fi_provider: str = "efa"
    nccl_proto: str = "simple"
    nccl_socket_ifname: str = "eth0"
    nccl_ib_disable: int = 1
    nccl_debug: str = "WARN"

########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

def get_checkpoint(training_args: DPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def main():

    parser = TrlParser((ModelConfig, ScriptArguments, DPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    #Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_download_location)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # MODEL CONFIGURATION
    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,  # Whether to trust the remote code, this also you to fine-tune custom architectures
        attn_implementation=model_args.attn_implementation,  # What attention implementation to use, defaults to flash_attention_2
        torch_dtype=(
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        ),  # What torch dtype to use, defaults to auto
        #use_cache=False if training_args.gradient_checkpointing else True,  # Whether
        low_cpu_mem_usage=(
            True
            if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))
            else None
        ),  # Reduces memory usage on CPU for loading the model
        
    )

    # Check which training method to use and if 4-bit quantization is needed
    if model_args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
        )
    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
    else:
        peft_config = None

    
    #Model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_download_location, 
        **model_kwargs)

    # Checks wether we use adapters for reference model or not
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(
              script_args.model_download_location, **model_kwargs
        )
    else:
        model_ref = None
    
    train_dataset = load_dataset("json", data_dir=script_args.dataset_local_location, split="train")
    
    #train_dataset = train_dataset.select_columns(["prompt", "chosen", "rejected"])
    train_dataset = train_dataset.select_columns(["chosen", "rejected"])
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        ref_model=model_ref,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )

    
    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    # last_checkpoint = get_checkpoint(training_args)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(
        #resume_from_checkpoint=last_checkpoint
    )
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()