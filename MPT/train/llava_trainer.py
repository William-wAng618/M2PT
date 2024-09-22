import os
import torch

from transformers import Trainer
from typing import Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

import contextlib
import copy
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
)

# isort: on

import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import Repository, create_repo
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.dependency_versions_check import dep_version_check
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    get_full_repo_name,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.utils import (
    ExplicitEnum,
    is_psutil_available,
    is_tf_available,
    is_torch_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_tpu_available,
    requires_backends,
)
PREFIX_CHECKPOINT_DIR = "checkpoint"
class ShardedDDPOption(ExplicitEnum):
    SIMPLE = "simple"
    ZERO_DP_2 = "zero_dp_2"
    ZERO_DP_3 = "zero_dp_3"
    OFFLOAD = "offload"
    AUTO_WRAP = "auto_wrap"
# from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
#                 from torch_xla.distributed.fsdp import checkpoint_module
#                 from torch_xla.distributed.fsdp.wrap import (
#                     size_based_auto_wrap_policy,
#                     transformer_auto_wrap_policy,
#                 )
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
def save_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, output_dir, optimizer_index=0):
    os.makedirs(output_dir, exist_ok=True)
    with FSDP.state_dict_type(
        model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
    ):
        optim_state = FSDP.optim_state_dict(model, optimizer)
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            if accelerator.process_index == 0:
                optim_state_name = (
                    f"{OPTIMIZER_NAME}.bin" if optimizer_index == 0 else f"{OPTIMIZER_NAME}_{optimizer_index}.bin"
                )
                output_optimizer_file = os.path.join(output_dir, optim_state_name)
                logger.info(f"Saving Optimizer state to {output_optimizer_file}")
                torch.save(optim_state, output_optimizer_file)
                logger.info(f"Optimizer state saved in {output_optimizer_file}")
        else:
            ckpt_dir = os.path.join(output_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f"Saving Optimizer state to {ckpt_dir}")
            dist_cp.save_state_dict(
                state_dict={"optimizer": optim_state},
                storage_writer=dist_cp.FileSystemWriter(ckpt_dir),
                planner=DefaultSavePlanner(),
            )
            logger.info(f"Optimizer state saved in {ckpt_dir}")

class LLaVATrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            print(f"in _save_checkpoint")
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    # def _save_checkpoint(self, model, trial, metrics=None):
    #     # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    #     # want to save except FullyShardedDDP.
    #     # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"
    #
    #     # Save model checkpoint
    #     checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-BEST"
    #
    #     if self.hp_search_backend is None and trial is None:
    #         self.store_flos()
    #
    #     run_dir = self._get_output_dir(trial=trial)
    #     exp_name = run_dir.split("/")[-1]
    #     checkpoint_folder = f"BEST-{exp_name}"
    #     output_dir = os.path.join(run_dir, checkpoint_folder)
    #     self.save_model(output_dir, _internal_call=True)
    #     if self.is_deepspeed_enabled:
    #         # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
    #         # config `stage3_gather_16bit_weights_on_model_save` is True
    #         self.model_wrapped.save_checkpoint(output_dir)
    #
    #     # Save optimizer and scheduler
    #     if self.sharded_ddp == ShardedDDPOption.SIMPLE:
    #         self.optimizer.consolidate_state_dict()
    #
    #     if self.fsdp or self.is_fsdp_enabled:
    #         if self.is_fsdp_enabled:
    #             save_fsdp_optimizer(
    #                 self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
    #             )
    #         else:
    #             # FSDP has a different interface for saving optimizer states.
    #             # Needs to be called on all ranks to gather all states.
    #             # full_optim_state_dict will be deprecated after Pytorch 2.2!
    #             full_osd = self.model.__class__.full_optim_state_dict(self.model, self.optimizer)
    #
    #     if is_torch_tpu_available():
    #         xm.rendezvous("saving_optimizer_states")
    #         xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
    #         with warnings.catch_warnings(record=True) as caught_warnings:
    #             xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
    #             reissue_pt_warnings(caught_warnings)
    #     elif is_sagemaker_mp_enabled():
    #         opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
    #         smp.barrier()
    #         if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
    #             smp.save(
    #                 opt_state_dict,
    #                 os.path.join(output_dir, OPTIMIZER_NAME),
    #                 partial=True,
    #                 v3=smp.state.cfg.shard_optimizer_state,
    #             )
    #         if self.args.should_save:
    #             with warnings.catch_warnings(record=True) as caught_warnings:
    #                 torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
    #             reissue_pt_warnings(caught_warnings)
    #             if self.do_grad_scaling:
    #                 torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
    #     elif self.args.should_save and not self.is_deepspeed_enabled:
    #         # deepspeed.save_checkpoint above saves model/optim/sched
    #         if self.fsdp and not self.is_fsdp_enabled:
    #             torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))
    #         else:
    #             torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
    #
    #         with warnings.catch_warnings(record=True) as caught_warnings:
    #             torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
    #         reissue_pt_warnings(caught_warnings)
    #         if self.do_grad_scaling:
    #             torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
    #
    #     # Determine the new best metric / best model checkpoint
    #     if metrics is not None and self.args.metric_for_best_model is not None:
    #         metric_to_check = self.args.metric_for_best_model
    #         if not metric_to_check.startswith("eval_"):
    #             metric_to_check = f"eval_{metric_to_check}"
    #         metric_value = metrics[metric_to_check]
    #
    #         operator = np.greater if self.args.greater_is_better else np.less
    #         if (
    #             self.state.best_metric is None
    #             or self.state.best_model_checkpoint is None
    #             or operator(metric_value, self.state.best_metric)
    #         ):
    #             self.state.best_metric = metric_value
    #             self.state.best_model_checkpoint = output_dir
    #
    #     # Save the Trainer state
    #     if self.args.should_save:
    #         self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
    #
    #     # Save RNG state in non-distributed training
    #     rng_states = {
    #         "python": random.getstate(),
    #         "numpy": np.random.get_state(),
    #         "cpu": torch.random.get_rng_state(),
    #     }
    #     if torch.cuda.is_available():
    #         if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
    #             # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
    #             rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
    #         else:
    #             rng_states["cuda"] = torch.cuda.random.get_rng_state()
    #
    #
    #     # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
    #     # not yet exist.
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     if self.args.world_size <= 1:
    #         torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
    #     else:
    #         torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))
    #
    #     if self.args.push_to_hub:
    #         self._push_from_checkpoint(output_dir)
    #
    #     # Maybe delete some older checkpoints.
    #     if self.args.should_save:
    #         self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
