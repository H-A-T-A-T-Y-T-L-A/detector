from abc import ABC , abstractmethod


class ImageProvider(ABC):

    name=""

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def dt(self):
        pass
