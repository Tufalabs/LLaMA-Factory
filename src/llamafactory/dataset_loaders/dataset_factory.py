from typing import Callable, ClassVar, TypeVar

from .base import BaseDatasetLoader
from .names import DatasetName


_BaseLoaderT = TypeVar("_BaseLoaderT", bound=type[BaseDatasetLoader])


class DatasetFactory:
    registered_loaders: ClassVar[dict[DatasetName, type[BaseDatasetLoader]]] = {}

    @classmethod
    def register_loader(cls, name: DatasetName) -> Callable[[_BaseLoaderT], _BaseLoaderT]:
        def wrapper(
            loader_cls: _BaseLoaderT,
        ) -> _BaseLoaderT:
            if name in cls.registered_loaders:
                raise RuntimeError(f"'{name}' is already registered")

            cls.registered_loaders[name] = loader_cls

            return loader_cls

        return wrapper

    @classmethod
    def get_loader(cls, name: DatasetName) -> BaseDatasetLoader:
        return cls.registered_loaders[name]()
