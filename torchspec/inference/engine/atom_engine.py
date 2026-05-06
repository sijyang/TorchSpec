# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
ATOM Ray actor engine for distributed deployment.

Uses ATOM's AsyncLLMEngine with RLHFModelRunner to capture intermediate
hidden states during prefill and store them directly to Mooncake via RDMA.

ATOM handles hidden states extraction inside its model forward pass
(via ``capture_hidden_state_layers`` in the model's ``forward()``),
so no external hooks or speculative config is needed.
"""

import os
import socket
from typing import Any

import ray
import torch

from torchspec.inference.engine.base import InferenceEngine
from torchspec.ray.ray_actor import RayActor
from torchspec.utils.logging import logger, setup_file_logging
from torchspec.utils.misc import get_default_eagle3_aux_layer_ids


class AtomEngine(InferenceEngine, RayActor):
    """Ray actor wrapper for ATOM AsyncLLMEngine.

    Captures intermediate hidden states during prefill and writes them
    to Mooncake store via RLHFModelRunner's hidden states extraction.
    """

    def __init__(
        self,
        args,
        rank: int,
        base_gpu_id: int | None = None,
        num_gpus_per_engine: int = 1,
        node_rank: int = 0,
        engine_group: int = 0,
    ):
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.num_gpus_per_engine = num_gpus_per_engine
        self.node_rank = node_rank
        self._engine = None
        self._mooncake_config = None
        self._hidden_size = None
        self.local_gpu_id = None

        setup_file_logging("inference", self.rank, group=engine_group)

    def init(self, mooncake_config=None) -> None:
        """Initialize the ATOM AsyncLLMEngine on the allocated GPU(s).

        Args:
            mooncake_config: MooncakeConfig object for distributed storage.
        """
        if self.base_gpu_id is not None:
            self.local_gpu_id = self.setup_gpu(self.base_gpu_id)
            logger.info(
                f"AtomEngine rank {self.rank}: base_gpu_id={self.base_gpu_id}, "
                f"using local GPU {self.local_gpu_id}"
            )

        self._mooncake_config = mooncake_config

        if mooncake_config is not None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = "localhost"
                logger.warning(
                    f"AtomEngine rank {self.rank}: failed to get local IP, using localhost"
                )

            mooncake_config.local_hostname = local_ip
            mooncake_config.export_env()

            from torchspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address,
                mooncake_config.metadata_server,
            )

        if self.args.aux_hidden_states_layers is not None:
            self.aux_hidden_state_layer_ids = list(self.args.aux_hidden_states_layers)
        else:
            self.aux_hidden_state_layer_ids = get_default_eagle3_aux_layer_ids(
                self.args.target_model_path
            )
            if self.rank == 0:
                logger.info(
                    f"Using default aux hidden state layer ids: "
                    f"{self.aux_hidden_state_layer_ids}"
                )

        # Pin ATOM workers to the correct physical GPUs
        if self.base_gpu_id is not None:
            gpu_ids = [
                str(self.base_gpu_id + i)
                for i in range(self.num_gpus_per_engine)
            ]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            logger.info(
                f"AtomEngine rank {self.rank}: set CUDA_VISIBLE_DEVICES="
                f"{os.environ['CUDA_VISIBLE_DEVICES']}"
            )

        tp_size = self.num_gpus_per_engine
        atom_config = {
            "enforce_eager": getattr(self.args, "atom_enforce_eager", True),
            "extra_args": getattr(self.args, "atom_extra_args", {}),
        }

        self._init_engine(tp_size, atom_config)

        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        hf_cfg = getattr(hf_cfg, "text_config", hf_cfg)
        self._hidden_size = hf_cfg.hidden_size

        # Configure hidden states extraction on all model runners
        mc_env = {}
        if mooncake_config is not None:
            import dataclasses

            mc_env = dataclasses.asdict(mooncake_config)

        self._engine.configure_hidden_states(
            aux_layer_ids=self.aux_hidden_state_layer_ids,
            mooncake_config=mc_env,
        )

        logger.info(
            f"AtomEngine rank {self.rank}: initialized from "
            f"{self.args.target_model_path} (tp_size={tp_size}, "
            f"aux_layers={self.aux_hidden_state_layer_ids}, "
            f"hidden_size={self._hidden_size})"
        )

    def _init_engine(self, tp_size: int, atom_config: dict) -> None:
        """Create the ATOM AsyncLLMEngine."""
        from atom.rollout.async_engine import AsyncLLMEngine

        enforce_eager = atom_config.get("enforce_eager", True)
        trust_remote_code = getattr(self.args, "trust_remote_code", True)

        engine_kwargs = {
            "tensor_parallel_size": tp_size,
            "enforce_eager": enforce_eager,
            "trust_remote_code": trust_remote_code,
        }

        max_seq_length = getattr(self.args, "max_seq_length", None)
        if max_seq_length:
            engine_kwargs["max_model_len"] = max_seq_length

        extra_args = atom_config.get("extra_args", {})
        if extra_args:
            engine_kwargs.update(extra_args)

        self._engine = AsyncLLMEngine(
            self.args.target_model_path,
            **engine_kwargs,
        )

    def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str | None] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate hidden states for training data.

        Hidden states are captured by ATOM's RLHFModelRunner during
        prefill and stored to Mooncake. Returns metadata for each
        request.
        """
        if self._engine is None:
            raise RuntimeError("AtomEngine not initialized. Call init() first.")

        assert input_ids_ref is not None, "input_ids_ref must not be None"

        if isinstance(input_ids_ref, ray.ObjectRef):
            input_ids_list = ray.get(input_ids_ref)
        else:
            input_ids_list = input_ids_ref

        batch_size = len(input_ids_list)

        if isinstance(data_id, str):
            data_ids = [f"{data_id}_{i}" for i in range(batch_size)]
        elif len(data_id) == batch_size:
            data_ids = list(data_id)
        else:
            raise ValueError(
                f"data_id length {len(data_id)} does not match "
                f"batch size {batch_size}"
            )

        token_lists = []
        for ids in input_ids_list:
            if isinstance(ids, torch.Tensor):
                if ids.dim() == 2 and ids.shape[0] == 1:
                    ids = ids.squeeze(0)
                token_lists.append(ids.tolist())
            else:
                token_lists.append(list(ids))

        logger.info(
            f"AtomEngine rank {self.rank}: processing {batch_size} requests, "
            f"data_ids={data_ids}, "
            f"seq_lens={[len(t) for t in token_lists]}"
        )

        self._engine.generate_hidden_states(
            input_ids_list=token_lists,
            data_ids=data_ids,
        )

        # Build packed_loss_mask_map for result assembly
        packed_loss_mask_map: dict[str, str | None] = {}
        if packed_loss_mask_list is not None:
            for i, did in enumerate(data_ids):
                if i < len(packed_loss_mask_list):
                    packed_loss_mask_map[did] = packed_loss_mask_list[i]

        num_aux = len(self.aux_hidden_state_layer_ids)
        results = []
        for i, did in enumerate(data_ids):
            seq_len = len(token_lists[i])
            result: dict[str, Any] = {
                "mooncake_key": did,
                "tensor_shapes": {
                    "hidden_states": [seq_len, num_aux * self._hidden_size],
                    "last_hidden_states": [seq_len, self._hidden_size],
                    "input_ids": [seq_len],
                },
                "tensor_dtypes": {
                    "hidden_states": "bfloat16",
                    "last_hidden_states": "bfloat16",
                    "input_ids": "int64",
                },
                "data_id": did,
                "seq_len": seq_len,
                "input_ids_list": token_lists[i],
            }

            packed_loss_mask = packed_loss_mask_map.get(did)
            if packed_loss_mask is not None:
                result["packed_loss_mask"] = packed_loss_mask

            results.append(result)

        logger.debug(
            f"AtomEngine rank {self.rank}: generated {len(results)} mooncake "
            f"results for data_ids={data_ids}"
        )
        return results

    def health_check(self, timeout: float = 5.0) -> bool:
        return self._engine is not None

    def shutdown(self) -> None:
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception as e:
                logger.warning(
                    f"AtomEngine rank {self.rank}: Error during shutdown: {e}"
                )
            finally:
                self._engine = None

        logger.info(f"AtomEngine rank {self.rank}: shutdown complete")

    def get_status(self) -> dict:
        return {
            "rank": self.rank,
            "initialized": self._engine is not None,
            "base_gpu_id": self.base_gpu_id,
            "hidden_size": self._hidden_size,
        }
