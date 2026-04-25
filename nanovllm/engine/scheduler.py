from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.prefill_chunk_size = config.prefill_chunk_size or config.max_num_batched_tokens
        self.max_decode_steps_before_prefill = config.max_decode_steps_before_prefill
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.decode_steps_since_prefill = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if seq.num_blocks > len(self.block_manager.blocks):
            raise ValueError("request prompt exceeds KV cache capacity")
        self.waiting.append(seq)

    def has_waiting(self):
        return bool(self.waiting)

    def has_running(self):
        return bool(self.running)

    def schedule(self) -> tuple[list[Sequence], bool]:
        should_prefill = self.waiting and (
            not self.running or self.decode_steps_since_prefill >= self.max_decode_steps_before_prefill
        )
        if not should_prefill:
            scheduled_seqs = self._schedule_decode()
            if scheduled_seqs:
                self.decode_steps_since_prefill += 1
                return scheduled_seqs, False

        scheduled_seqs = self._schedule_prefill()
        if scheduled_seqs:
            self.decode_steps_since_prefill = 0
            return scheduled_seqs, True

        scheduled_seqs = self._schedule_decode()
        if scheduled_seqs:
            self.decode_steps_since_prefill += 1
            return scheduled_seqs, False

        raise RuntimeError("no schedulable requests")

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0
        waiting_seqs = len(self.waiting)
        for _ in range(waiting_seqs):
            if len(scheduled_seqs) == self.max_num_seqs:
                break
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            seq = self.waiting.popleft()
            if not seq.block_table and not self.block_manager.can_allocate(seq):
                self.waiting.append(seq)
                continue
            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
            if not seq.block_table:
                self.block_manager.allocate(seq)
            seq.num_scheduled_tokens = min(num_tokens, remaining, self.prefill_chunk_size)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                self.waiting.append(seq)
            scheduled_seqs.append(seq)
            num_batched_tokens += seq.num_scheduled_tokens
        return scheduled_seqs

    def _schedule_decode(self) -> list[Sequence]:
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            return scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def abort(self, seq: Sequence):
        if seq.status == SequenceStatus.WAITING:
            self.waiting.remove(seq)
        elif seq.status == SequenceStatus.RUNNING:
            self.running.remove(seq)
        else:
            return False
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.ABORTED
        seq.finish_reason = "abort"
        seq.num_scheduled_tokens = 0
        return True

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        events = []
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:    # chunked prefill or re prefill after preemption
                    seq.num_scheduled_tokens = 0
                    continue
            seq.append_token(token_id)
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0
            finish_reason = None
            if not seq.ignore_eos and token_id == self.eos:
                finish_reason = "stop"
            elif seq.num_completion_tokens == seq.max_tokens:
                finish_reason = "length"
            if finish_reason is not None:
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = finish_reason
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            events.append((seq, token_id, finish_reason))
        return events
