"""
DetokenizerProcess -- subprocess that converts token IDs back to text.

Receives ``BatchTokenIDOut``-style dicts from the SchedulerProcess,
detokenizes them, and forwards the decoded strings to the
RequestResponseProcess.
"""

import logging
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional

import zmq

from pymllm.orchestrator.ipc_utils import create_zmq_socket, setup_subprocess_logging

logger = logging.getLogger(__name__)


class DetokenizerProcess:
    """Runs inside a subprocess.  Detokenizes finished outputs."""

    def __init__(
        self,
        recv_from_scheduler_addr: str,
        send_to_rr_addr: str,
        tokenizer_cfg: Optional[Dict[str, Any]] = None,
    ):
        self._recv_from_scheduler_addr = recv_from_scheduler_addr
        self._send_to_rr_addr = send_to_rr_addr
        self._tokenizer_cfg = tokenizer_cfg or {}

        self._zmq_ctx: Optional[zmq.Context] = None
        self._recv_from_scheduler: Optional[zmq.Socket] = None
        self._send_to_rr: Optional[zmq.Socket] = None

        self._tokenizer = None
        self._reasoning_parser = self._tokenizer_cfg.get("reasoning_parser")
        self._qwen_think_end_token_ids: List[int] = []
        # Track previous decoded text per rid for incremental (delta) output
        self._rid_to_prev_text: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_sockets(self) -> None:
        self._zmq_ctx = zmq.Context()
        self._recv_from_scheduler = create_zmq_socket(
            self._zmq_ctx,
            zmq.PULL,
            self._recv_from_scheduler_addr,
            bind=False,
        )
        self._send_to_rr = create_zmq_socket(
            self._zmq_ctx,
            zmq.PUSH,
            self._send_to_rr_addr,
            bind=False,
        )

    def init_tokenizer(self) -> None:
        """Load the tokenizer from the configured path."""
        tokenizer_path = self._tokenizer_cfg.get("tokenizer_path")
        if tokenizer_path is None:
            logger.warning(
                "No tokenizer_path in tokenizer_cfg; detokenization disabled"
            )
            return

        from transformers import AutoTokenizer

        trust_remote_code = self._tokenizer_cfg.get("trust_remote_code", False)
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
        self._qwen_think_end_token_ids = self._resolve_qwen_think_end_token_ids()
        logger.info("Detokenizer loaded tokenizer from %s", tokenizer_path)
        if self._uses_qwen_reasoning_parser():
            logger.info(
                "Qwen reasoning token-id parser enabled; </think> token ids: %s",
                self._qwen_think_end_token_ids,
            )

    def _resolve_qwen_think_end_token_ids(self) -> List[int]:
        if self._tokenizer is None:
            return []

        token_ids: List[int] = []
        unk_id = getattr(self._tokenizer, "unk_token_id", None)

        convert = getattr(self._tokenizer, "convert_tokens_to_ids", None)
        if convert is not None:
            try:
                token_id = convert("</think>")
            except Exception:
                token_id = None
            if isinstance(token_id, int) and token_id != unk_id:
                token_ids.append(token_id)

        encode = getattr(self._tokenizer, "encode", None)
        if encode is not None:
            try:
                encoded = encode("</think>", add_special_tokens=False)
            except Exception:
                encoded = []
            if isinstance(encoded, list) and len(encoded) == 1:
                token_id = encoded[0]
                if isinstance(token_id, int) and token_id != unk_id:
                    token_ids.append(token_id)

        # Qwen3 runner in this repo uses 151668 as </think>; Qwen3.5 uses
        # 248069 in the local serving setup.
        token_ids.append(151668)
        token_ids.append(248069)
        return list(dict.fromkeys(token_ids))

    def _uses_qwen_reasoning_parser(self) -> bool:
        return self._reasoning_parser in {"qwen3", "qwen3-thinking"}

    def _find_last_think_end(self, output_ids: List[int]) -> Optional[int]:
        if not self._qwen_think_end_token_ids:
            return None

        end_ids = set(self._qwen_think_end_token_ids)
        for index in range(len(output_ids) - 1, -1, -1):
            if output_ids[index] in end_ids:
                return index
        return None

    def _decode_visible_text(
        self,
        output_ids: List[int],
        *,
        skip_special_tokens: bool,
        finished_reason: Optional[str],
    ) -> tuple[str, bool, int, Optional[int]]:
        if self._tokenizer is None:
            return "", False, 0, None

        if not self._uses_qwen_reasoning_parser():
            text = self._tokenizer.decode(
                output_ids,
                skip_special_tokens=skip_special_tokens,
            )
            return (
                text,
                False,
                len(output_ids),
                None,
            )

        think_end_index = self._find_last_think_end(output_ids)
        if think_end_index is None:
            return (
                self._tokenizer.decode(
                    output_ids,
                    skip_special_tokens=skip_special_tokens,
                ),
                False,
                len(output_ids),
                None,
            )

        content_ids = output_ids[think_end_index + 1 :]
        return (
            self._tokenizer.decode(
                content_ids,
                skip_special_tokens=True,
            ),
            True,
            len(content_ids),
            output_ids[think_end_index],
        )

    def event_loop(self) -> None:
        """Infinite loop: recv token IDs -> detokenize -> send text to RR."""
        logger.info("DetokenizerProcess event loop started")
        while True:
            token_id_out = self._recv_from_scheduler.recv_pyobj()
            results = self._detokenize(token_id_out)
            for result in results:
                self._send_to_rr.send_pyobj(result)

    # ------------------------------------------------------------------
    # Detokenization
    # ------------------------------------------------------------------

    def _detokenize(self, token_id_out: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert token IDs to text and fan out one result per rid.

        The scheduler sends a batch dict with parallel lists keyed by
        ``"rids"``, ``"output_ids"``, ``"finished_reasons"``, etc.
        This method decodes each rid's output_ids and produces one result
        dict per rid with keys ``"rid"`` (singular) and ``"finished"``
        (bool) as expected by ``RequestResponseProcess._recv_loop``.
        """
        rids: List[str] = token_id_out.get("rids", [])
        output_ids: List[int] = token_id_out.get("output_ids", [])
        finished_reasons: List[Optional[str]] = token_id_out.get("finished_reasons", [])

        # NOTE: The scheduler currently sends one rid per message.  The shared
        # output_ids list is the complete output for that single rid.  If
        # batched sending is ever added, each rid will need its own output_ids.
        if len(rids) > 1:
            logger.warning(
                "Detokenizer received %d rids in one message; "
                "output_ids are shared -- results may be incorrect",
                len(rids),
            )
        decode_ids: List[int] = token_id_out.get("decode_ids", [])
        skip_special_tokens_list: List[bool] = token_id_out.get(
            "skip_special_tokens", []
        )
        prompt_tokens_list: List[int] = token_id_out.get("prompt_tokens", [])
        completion_tokens_list: List[int] = token_id_out.get("completion_tokens", [])
        vit_prefill_ms_list = token_id_out.get("vit_prefill_ms", [])
        vit_prefill_tokens_list = token_id_out.get("vit_prefill_tokens", [])
        llm_prefill_ms_list = token_id_out.get("llm_prefill_ms", [])
        llm_decode_ms_list = token_id_out.get("llm_decode_ms", [])

        results: List[Dict[str, Any]] = []

        for i, rid in enumerate(rids):
            finished_reason = finished_reasons[i] if i < len(finished_reasons) else None
            is_finished = finished_reason is not None
            skip_special = (
                skip_special_tokens_list[i]
                if i < len(skip_special_tokens_list)
                else True
            )
            prompt_tokens = prompt_tokens_list[i] if i < len(prompt_tokens_list) else 0
            completion_tokens = (
                completion_tokens_list[i] if i < len(completion_tokens_list) else 0
            )
            vit_prefill_ms = (
                vit_prefill_ms_list[i] if i < len(vit_prefill_ms_list) else None
            )
            vit_prefill_tokens = (
                vit_prefill_tokens_list[i] if i < len(vit_prefill_tokens_list) else None
            )
            llm_prefill_ms = (
                llm_prefill_ms_list[i] if i < len(llm_prefill_ms_list) else None
            )
            llm_decode_ms = (
                llm_decode_ms_list[i] if i < len(llm_decode_ms_list) else None
            )

            (
                text,
                reasoning_closed,
                visible_token_count,
                think_end_token_id,
            ) = self._decode_visible_text(
                output_ids,
                skip_special_tokens=skip_special,
                finished_reason=finished_reason,
            )
            if (
                self._uses_qwen_reasoning_parser()
                and is_finished
                and not reasoning_closed
                and not text
                and self._tokenizer is not None
            ):
                tail_ids = list(output_ids[-50:])
                convert = getattr(self._tokenizer, "convert_ids_to_tokens", None)
                if convert is not None:
                    try:
                        tail_tokens = convert(tail_ids)
                    except Exception:
                        tail_tokens = []
                else:
                    tail_tokens = []
                logger.warning(
                    "Qwen reasoning did not close; finish=%s, "
                    "completion_tokens=%s, tail_ids=%s, tail_tokens=%s",
                    finished_reason,
                    completion_tokens,
                    tail_ids,
                    tail_tokens,
                )

            # Compute incremental delta by diffing against previous text
            prev_text = self._rid_to_prev_text.get(rid, "")
            delta_text = text[len(prev_text):]
            self._rid_to_prev_text[rid] = text

            # Clean up tracking when request finishes
            if is_finished:
                self._rid_to_prev_text.pop(rid, None)

            result: Dict[str, Any] = {
                "rid": rid,
                "text": text,
                "delta": delta_text,
                "output_token_ids": list(output_ids),
                "finished": is_finished,
                "finished_reason": finished_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_closed": reasoning_closed,
                "visible_token_count": visible_token_count,
                "think_end_token_id": think_end_token_id,
            }
            if vit_prefill_ms is not None:
                result["vit_prefill_ms"] = vit_prefill_ms
            if vit_prefill_tokens is not None:
                result["vit_prefill_tokens"] = vit_prefill_tokens
            if llm_prefill_ms is not None:
                result["llm_prefill_ms"] = llm_prefill_ms
            if llm_decode_ms is not None:
                result["llm_decode_ms"] = llm_decode_ms
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._recv_from_scheduler is not None:
            self._recv_from_scheduler.close()
        if self._send_to_rr is not None:
            self._send_to_rr.close()
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()


def run_detokenizer_process(
    recv_from_scheduler_addr: str,
    send_to_rr_addr: str,
    pipe_writer: Connection,
    tokenizer_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Entry point for ``torch.multiprocessing.Process(target=...)``."""
    setup_subprocess_logging((tokenizer_cfg or {}).get("log_level", "info"))

    # Limit CPU threads — detokenizer doesn't need PyTorch parallelism.
    import torch
    torch.set_num_threads(1)

    proc = DetokenizerProcess(
        recv_from_scheduler_addr,
        send_to_rr_addr,
        tokenizer_cfg=tokenizer_cfg,
    )
    proc.init_sockets()
    proc.init_tokenizer()

    pipe_writer.send({"status": "ready", "process": "detokenizer"})
    pipe_writer.close()

    try:
        proc.event_loop()
    except KeyboardInterrupt:
        pass
    finally:
        proc.shutdown()
