import atexit
from dataclasses import dataclass, fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


@dataclass(slots=True)
class RequestEvent:
    request_id: int | str
    seq_id: int
    token_id: int
    finish_reason: str | None = None


@dataclass(slots=True)
class RequestOutput:
    request_id: int | str
    seq_id: int
    token_ids: list[int]
    finish_reason: str


@dataclass(slots=True)
class StepStats:
    is_prefill: bool
    scheduled_tokens: int
    num_waiting: int
    num_running: int
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    is_mixed: bool = False


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.requests: dict[int | str, Sequence] = {}
        self.completed: dict[int | str, RequestOutput] = {}
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def submit_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: int | str | None = None,
    ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        if request_id is not None:
            seq.request_id = request_id
        if seq.request_id in self.requests or seq.request_id in self.completed:
            raise ValueError(f"duplicate request id: {seq.request_id!r}")
        self.scheduler.add(seq)
        self.requests[seq.request_id] = seq
        return seq.request_id

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: int | str | None = None,
    ):
        return self.submit_request(prompt, sampling_params, request_id)

    def abort_request(self, request_id: int | str):
        seq = self.requests.get(request_id)
        if seq is None or not self.scheduler.abort(seq):
            return None
        return self._finalize_request(seq)

    def pop_completed(self, request_id: int | str):
        return self.completed.pop(request_id, None)

    def has_pending_requests(self):
        return bool(self.requests)

    def _finalize_request(self, seq: Sequence):
        output = RequestOutput(
            request_id=seq.request_id,
            seq_id=seq.seq_id,
            token_ids=seq.completion_token_ids,
            finish_reason=seq.finish_reason or "abort",
        )
        self.requests.pop(seq.request_id, None)
        self.completed[seq.request_id] = output
        return output

    def step(self):
        if self.is_finished():
            return [], StepStats(False, 0, 0, 0)
        batch = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", batch)
        scheduler_events = self.scheduler.postprocess(batch, token_ids)
        events = []
        for seq, token_id, finish_reason in scheduler_events:
            if finish_reason is not None:
                self._finalize_request(seq)
            events.append(RequestEvent(seq.request_id, seq.seq_id, token_id, finish_reason))
        stats = StepStats(
            is_prefill=batch.is_prefill_only,
            scheduled_tokens=batch.num_tokens,
            num_waiting=len(self.scheduler.waiting),
            num_running=len(self.scheduler.running),
            num_prefill_tokens=batch.num_prefill_tokens,
            num_decode_tokens=batch.num_decode_tokens,
            is_mixed=batch.is_mixed,
        )
        return events, stats

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        request_ids = [self.submit_request(prompt, sp) for prompt, sp in zip(prompts, sampling_params)]
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            events, stats = self.step()
            step_time = perf_counter() - t
            if stats.num_prefill_tokens:
                prefill_throughput = stats.num_prefill_tokens / step_time
            if stats.num_decode_tokens:
                decode_throughput = stats.num_decode_tokens / step_time
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for event in events:
                if event.finish_reason is None:
                    continue
                output = self.pop_completed(event.request_id)
                outputs[event.request_id] = output.token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[request_id] for request_id in request_ids]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
