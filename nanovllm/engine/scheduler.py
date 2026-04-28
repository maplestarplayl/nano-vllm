from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass(slots=True)
class ScheduleEntry:
    seq: Sequence
    is_prefill: bool

    @property
    def num_tokens(self):
        return self.seq.num_scheduled_tokens


@dataclass(slots=True)
class ScheduleBatch:
    entries: list[ScheduleEntry]

    @property
    def seqs(self):
        return [entry.seq for entry in self.entries]

    @property
    def has_prefill(self):
        return any(entry.is_prefill for entry in self.entries)

    @property
    def has_decode(self):
        return any(not entry.is_prefill for entry in self.entries)

    @property
    def is_prefill_only(self):
        return self.has_prefill and not self.has_decode

    @property
    def is_decode_only(self):
        return self.has_decode and not self.has_prefill

    @property
    def is_mixed(self):
        return self.has_prefill and self.has_decode

    @property
    def num_prefill_tokens(self):
        return sum(entry.num_tokens for entry in self.entries if entry.is_prefill)

    @property
    def num_decode_tokens(self):
        return sum(entry.num_tokens for entry in self.entries if not entry.is_prefill)

    @property
    def num_tokens(self):
        return self.num_prefill_tokens + self.num_decode_tokens


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

    def schedule(self) -> ScheduleBatch:
        entries = []
        token_budget = self.max_num_batched_tokens
        seq_budget = self.max_num_seqs
        should_schedule_prefill = self.waiting and (
            not self.running or self.decode_steps_since_prefill >= self.max_decode_steps_before_prefill - 1
        )

        if self.running:
            decode_seq_budget = seq_budget
            if should_schedule_prefill and seq_budget > 1:
                decode_seq_budget -= 1
            decode_token_budget = token_budget
            if should_schedule_prefill and token_budget > 1:
                decode_token_budget -= 1
            decode_entries = self._schedule_decode(decode_seq_budget, decode_token_budget)
            entries.extend(decode_entries)
            seq_budget -= len(decode_entries)
            token_budget -= sum(entry.num_tokens for entry in decode_entries)

        if should_schedule_prefill and seq_budget > 0 and token_budget > 0:
            prefill_entries = self._schedule_prefill(seq_budget, token_budget)
            entries.extend(prefill_entries)
            seq_budget -= len(prefill_entries)
            token_budget -= sum(entry.num_tokens for entry in prefill_entries)

        if not entries and self.running:
            entries.extend(self._schedule_decode(seq_budget, token_budget))

        if entries:
            batch = ScheduleBatch(entries)
            self.decode_steps_since_prefill = 0 if batch.has_prefill else self.decode_steps_since_prefill + 1
            return batch

        raise RuntimeError("no schedulable requests")

    def _schedule_prefill(self, max_seqs: int, max_tokens: int) -> list[ScheduleEntry]:
        entries = []
        num_batched_tokens = 0
        waiting_seqs = len(self.waiting)
        for _ in range(waiting_seqs):
            if len(entries) == max_seqs:
                break
            remaining = max_tokens - num_batched_tokens
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
            entries.append(ScheduleEntry(seq, True))
            num_batched_tokens += seq.num_scheduled_tokens
        return entries

    def _schedule_decode(self, max_seqs: int, max_tokens: int) -> list[ScheduleEntry]:
        entries = []
        while self.running and len(entries) < max_seqs and len(entries) < max_tokens:
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
                entries.append(ScheduleEntry(seq, False))
        if not entries:
            return entries
        self.running.extendleft(entry.seq for entry in reversed(entries))
        return entries

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

    def postprocess(self, batch: ScheduleBatch, token_ids: list[int]):
        events = []
        for entry, token_id in zip(batch.entries, token_ids):
            seq = entry.seq
            if entry.is_prefill:
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
