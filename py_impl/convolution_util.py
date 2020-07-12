from enum import Enum, auto
from math import floor


class ChannelPosition(Enum):
    FIRST = auto()
    LAST = auto()
    IMPLICIT = auto()

    def __str__(self):
        if self is ChannelPosition.FIRST:
            return "ChannelPosition::FIRST"
        elif self is ChannelPosition.LAST:
            return "ChannelPosition::LAST"
        elif self is ChannelPosition.IMPLICIT:
            return "ChannelPosition::IMPLICIT"
        else:
            return "Unknown ChannelPosition"

    def __repr__(self):
        return str(self)


class KernelOptions:
    def __init__(self,
                 padding=0,
                 stride=1,
                 dilation=1,
                 channel_position=ChannelPosition.LAST):
        self._padding = padding
        self._stride = stride
        self._dilation = dilation
        self._channel_position = channel_position

    @property
    def padding(self):
        return self._padding

    @property
    def padding_total(self):
        return self.padding_begin + self.padding_end

    @property
    def padding_begin(self):
        return self.padding

    @property
    def padding_end(self):
        return self.padding

    @property
    def stride(self):
        return self._stride

    @property
    def dilation(self):
        return self._dilation

    @property
    def channel_position(self):
        return self._channel_position

    def __str__(self):
        padding = self.padding_begin, self.padding_end, self.padding_total
        return f"padding (Begin, End, Total)={padding}," \
               f" stride={self.stride}," \
               f" dilation={self.dilation}," \
               f" channel_position={self.channel_position})"


def calculate_output_size(input_size, kernel_size, options):
    non_strided_size = input_size + options.padding_total - options.dilation * (kernel_size - 1) - 1
    return int(floor(non_strided_size / options.stride + 1))
