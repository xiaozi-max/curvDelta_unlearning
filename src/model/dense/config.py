from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import TaskType


DENSE_PEFT_TYPE = "DENSE"


@dataclass
class DenseConfig(PeftConfig):
    """PEFT configuration for a full-rank additive weight adapter."""

    target_modules: Optional[Union[list[str], str]] = field(default=None)
    exclude_modules: Optional[Union[list[str], str]] = field(default=None)
    layers_to_transform: Optional[Union[list[int], int]] = field(default=None)
    layers_pattern: Optional[Union[list[str], str]] = field(default=None)
    bias: str = field(default="none")
    modules_to_save: Optional[list[str]] = field(default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.peft_type = DENSE_PEFT_TYPE

        if self.task_type is not None and self.task_type != TaskType.CAUSAL_LM:
            raise ValueError(
                "Dense adapters currently support only task_type='CAUSAL_LM'."
            )
        if self.bias != "none":
            raise ValueError("Dense adapters do not support training bias parameters.")
        if self.modules_to_save:
            raise ValueError("Dense adapters do not support modules_to_save.")

        self.target_modules = (
            set(self.target_modules)
            if isinstance(self.target_modules, list)
            else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules)
            if isinstance(self.exclude_modules, list)
            else self.exclude_modules
        )

        if (
            isinstance(self.target_modules, str)
            and self.layers_to_transform is not None
        ):
            raise ValueError(
                "layers_to_transform cannot be used when target_modules is a regex string."
            )
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError(
                "layers_pattern cannot be used when target_modules is a regex string."
            )
        if self.layers_pattern and self.layers_to_transform is None:
            raise ValueError(
                "layers_pattern requires layers_to_transform to be specified."
            )
