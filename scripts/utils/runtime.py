from __future__ import annotations

import logging
import queue
import threading

import torch


class AverageMeter:
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class BackgroundBatchBuffer:
    """Background prefetch buffer with an optional CUDA transfer stream."""

    _SENTINEL = object()

    def __init__(
        self,
        dataloader,
        device,
        buffer_size: int = 5,
        convert_fn=None,
        warmup_fraction: float = 0.8,
        on_epoch_advance=None,
    ):
        self._dataloader = dataloader
        self._device = device
        self._buffer_size = buffer_size
        self._convert_fn = convert_fn
        self._queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._error = None
        self._warmup_target = max(1, int(buffer_size * warmup_fraction))
        self._warmup_done = threading.Event()
        # Only create a CUDA stream when the target device is CUDA; guarding on
        # torch.cuda.is_available() alone would make a CPU/MPS run on a
        # CUDA-capable host raise on torch.cuda.Stream(device='cpu').
        self._on_cuda_device = (
            torch.cuda.is_available()
            and device is not None
            and torch.device(device).type == "cuda"
        )
        self._transfer_stream = (
            torch.cuda.Stream(device=device) if self._on_cuda_device else None
        )
        self._on_epoch_advance = on_epoch_advance
        self._epoch = 0
        self._thread = threading.Thread(target=self._fill_loop, daemon=True)
        self._thread.start()

    def _fill_loop(self) -> None:
        try:
            # Only call set_device when the target device is CUDA.
            if self._on_cuda_device:
                dev = self._device
                if isinstance(dev, torch.device) and dev.index is None:
                    dev = torch.device("cuda", torch.cuda.current_device())
                torch.cuda.set_device(dev)

            dl_iter = iter(self._dataloader)
            while not self._stop_event.is_set():
                try:
                    batch = next(dl_iter)
                except StopIteration:
                    self._epoch += 1
                    if self._on_epoch_advance is not None:
                        try:
                            self._on_epoch_advance(self._epoch)
                        except Exception as cb_err:
                            logging.warning(
                                "on_epoch_advance callback failed at epoch=%s: %s",
                                self._epoch,
                                cb_err,
                            )
                    dl_iter = iter(self._dataloader)
                    continue

                if self._transfer_stream is not None:
                    with torch.cuda.stream(self._transfer_stream):
                        if self._convert_fn is not None:
                            batch = self._convert_fn(batch, self._device)
                    event = self._transfer_stream.record_event()
                else:
                    if self._convert_fn is not None:
                        batch = self._convert_fn(batch, self._device)
                    event = None

                item = (batch, event)
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if not self._warmup_done.is_set() and self._queue.qsize() >= self._warmup_target:
                    self._warmup_done.set()
        except Exception as e:
            logging.exception("BackgroundBatchBuffer producer thread died: %s", e)
            self._error = e
            try:
                self._queue.put((self._SENTINEL, None), timeout=5.0)
            except queue.Full:
                pass

    def wait_for_warmup(self, timeout: int = 300) -> int:
        if self._warmup_done.wait(timeout=timeout):
            return self._queue.qsize()
        logging.warning(
            "Buffer warmup timeout (%ss), starting with buf=%s/%s (target=%s)",
            timeout,
            self._queue.qsize(),
            self._buffer_size,
            self._warmup_target,
        )
        return self._queue.qsize()

    def get(self, timeout: int = 1800):
        if self._error is not None:
            raise RuntimeError(
                f"BackgroundBatchBuffer background thread terminated with error: {self._error}"
            ) from self._error
        try:
            batch, event = self._queue.get(timeout=timeout)
        except queue.Empty:
            if self._error is not None:
                raise RuntimeError(
                    f"BackgroundBatchBuffer background thread terminated with error: {self._error}"
                ) from self._error
            raise RuntimeError(
                f"BackgroundBatchBuffer.get() timed out after {timeout}s "
                "(no batch produced; check producer logs for stack trace)"
            )
        if batch is self._SENTINEL:
            raise RuntimeError(
                f"BackgroundBatchBuffer background thread terminated with error: {self._error}"
            ) from self._error
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        return batch

    def stop(self) -> None:
        self._stop_event.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break
        self._thread.join(timeout=5.0)

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def epoch(self) -> int:
        return self._epoch


def format_time(seconds) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def set_seed(seed) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_epoch_recursive(ds, epoch: int) -> int:
    if ds is None:
        return 0
    n_called = 0
    seen: set[int] = set()

    def _walk(obj) -> None:
        nonlocal n_called
        if obj is None:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        fn = getattr(obj, "set_epoch", None)
        if callable(fn):
            try:
                fn(int(epoch))
                n_called += 1
            except Exception as e:
                logging.warning("set_epoch on %s failed: %s", type(obj).__name__, e)

        for attr in ("_datasets", "datasets", "dataset", "adapter", "_adapter"):
            child = getattr(obj, attr, None)
            if child is None:
                continue
            if isinstance(child, (list, tuple)):
                for c in child:
                    _walk(c)
            else:
                _walk(child)

    _walk(ds)
    return n_called


def _set_sampler_start_recursive(obj, start_batch_index: int, *, epoch: int | None = None) -> int:
    seen = set()
    n_called = 0

    def _walk(item) -> None:
        nonlocal n_called
        if item is None or id(item) in seen:
            return
        seen.add(id(item))
        if epoch is not None:
            epoch_fn = getattr(item, "set_epoch", None)
            if callable(epoch_fn):
                try:
                    epoch_fn(int(epoch))
                except Exception as e:
                    logging.warning("set_epoch on %s failed: %s", type(item).__name__, e)
        fn = getattr(item, "set_start_batch_index", None)
        if callable(fn):
            try:
                fn(int(start_batch_index))
                n_called += 1
            except Exception as e:
                logging.warning(
                    "set_start_batch_index on %s failed: %s",
                    type(item).__name__,
                    e,
                )
        for attr in ("batch_sampler", "sampler", "dataloader", "_dataloader", "dataset"):
            child = getattr(item, attr, None)
            if child is not None:
                _walk(child)

    _walk(obj)
    return n_called


def setup_logging(accelerator, log_file=None) -> None:
    try:
        from utils.utils import init_logging

        init_logging(
            log_file=log_file,
            console_level="INFO",
            file_level="INFO",
            accelerator=accelerator,
        )
    except Exception as e:
        log_level = logging.INFO if accelerator.is_main_process else logging.WARNING
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=log_level,
        )
        logging.warning("init_logging unavailable, using basicConfig. Reason: %s", e)

    try:
        from utils.logging_utils import install_dedupe_filter

        install_dedupe_filter()
    except Exception as e:
        logging.debug("DedupeFilter install skipped: %s", e)
