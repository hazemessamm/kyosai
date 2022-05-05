import time
import sys
import math

class ProgressBar:
    def __init__(self, total, width=30, metrics=None, unit='step'):
        self._total = total
        self._width = width
        self._size_per_percent = width / total
        self._done = 0
        self._start_time = time.time()
        self._new_line = False
        self._unit = unit
        self._msg_format = '\r{epoch}/{total} [{bar}] {percent:.2f}% '


    def update(self, epoch, **kwargs):
        percent = math.ceil((epoch / self._total) * 100)
        self._done += self._size_per_percent
        bar = ''.join(['=' * int(self._done),  '>', '-' * int(self._width - self._done + self._size_per_percent)])
        # time_per_step = time.time()'
        bar = self._msg_format.format(epoch=epoch, total=self._total, bar=bar, percent=percent)
        bar = ''.join([bar, ('{}: {:.5f} ' * len(kwargs)).format(*(y for kv in kwargs.items() for y in kv))])
        self._print_message(bar, self._new_line)
        self._new_line = False

    def _print_message(self, msg, new_line=False):
        if new_line:
            msg += '\n'
        sys.stdout.write(msg)

    def reset(self):
        self._done = 0
        self._new_line = True