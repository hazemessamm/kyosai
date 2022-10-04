import math
import sys
import time


class ProgressBar:
    def __init__(self, total, width=30, metrics=None, unit="step"):
        self.total = total
        self.width = width
        self._size_per_percent = width / total
        self._done = 0
        self._start_time = time.time()
        self._new_line = False
        self._printed_epoch = False
        self._first_call = True
        self._unit = unit

    def update(self, epoch, step, **kwargs):
        self.print_epoch(epoch)
        percent = math.ceil((step / self.total) * 100)
        self._done += self._size_per_percent
        bar = (
            ("=" * int(self._done))
            + ">"
            + ("." * int(self.width - int(self._done) + self._size_per_percent))
        )

        bar = "\r{percent}% [{bar}] {step}/{total} ".format(
            percent=percent, bar=bar, step=step, total=self.total
        )
        bar += ("{}: {:.5f} " * len(kwargs)).format(
            *(y for kv in kwargs.items() for y in kv)
        )
        self._print_message(bar)
        self._new_line = False

    def print_epoch(self, epoch):
        if not self._printed_epoch:
            if self._first_call:
                sys.stdout.write(f"Epoch: {epoch}\n")
                self._first_call = False
            else:
                sys.stdout.write(f"\nEpoch: {epoch}")
            self._printed_epoch = True

    def _print_message(self, msg):
        if self._new_line:
            sys.stdout.write("\n")
            sys.stdout.write(msg)
        else:
            sys.stdout.write(msg)

    def reset(self):
        self._done = 0
        self._new_line = True
        self._printed_epoch = False
        sys.stdout.flush()
