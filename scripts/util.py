from io import IOBase, TextIOBase
from sys import stdout
from time import time, sleep
from datetime import datetime


def time_in_millis():
    return int(round(time() * 1000))


class TimeMeasure:
    def __init__(self,
                 enter_message=f'{datetime.now().strftime("%d.%m.%Y %H:%M:%S")}',
                 exit_message="{}"
                 ):
        self._enter_message = enter_message
        self._exit_message = exit_message
        self._start = 0

    def __enter__(self):
        print(self._enter_message)
        self._start = time_in_millis()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        delta = time_in_millis() - self._start
        print(self._exit_message.format(format_millis(delta)))


def format_millis(millis):
    (hours, minutes) = divmod(millis, 3600000)
    (minutes, seconds) = divmod(minutes, 60000)
    (seconds, millis) = divmod(seconds, 1000)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:04}"


def string_framed_line(title, length=120, orientation='^', style='-'):
    lines = list()
    length -= 2
    if style == '=':
        lines.append('╔' + '═' * length + '╗')
        lines.append(('║ {:' + f'{orientation}{length - 2}' + '} ║').format(title))
        lines.append('╚' + '═' * length + '╝')
    elif style == '-':
        lines.append('┌' + '─' * length + '┐')
        lines.append(('│ {:' + f'{orientation}{length - 2}' + '} │').format(title))
        lines.append('└' + '─' * length + '┘')
    else:
        raise ValueError(f"Unknown frame style '{style}' (console.py: string_framed_line)")
    return lines
