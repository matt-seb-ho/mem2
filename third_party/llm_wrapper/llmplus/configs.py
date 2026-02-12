from dataclasses import asdict, dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .model_registry import ModelMeta


@dataclass
class GenerationConfig:
    n: int | list[int] = 1
    temperature: float = 0.3
    max_tokens: int = 1024
    top_p: float = 1.0
    seed: int | None = None
    extra_kwargs: dict[str, Any] | None = None

    # calling options for wrapper operation (not forwarded to API request)
    expand_multi: bool | None = field(default=None, metadata={"forward": False})
    batch_size: int = field(default=16, metadata={"forward": False})
    ignore_cache: bool = field(default=False, metadata={"forward": False})

    def __post_init__(self):
        # hydra initialization sometimes leads to extra_kwargs being a DictConfig
        # orjson cannot serialize DictConfig
        if not self.extra_kwargs:
            return
        if isinstance(self.extra_kwargs, DictConfig):
            self.extra_kwargs = OmegaConf.to_container(self.extra_kwargs, resolve=True)
        else:
            for k, v in self.extra_kwargs.items():
                self.extra_kwargs[k] = self._preprocess_extra_kwarg(v)

    def override(self, **kwargs) -> "GenerationConfig":
        """Create a new instance with updated values."""
        arg_dict = asdict(self)
        extra_kwargs = arg_dict.pop("extra_kwargs", {}) or {}
        for k, v in kwargs.items():
            if k in arg_dict:
                arg_dict[k] = v
            else:
                extra_kwargs[k] = self._preprocess_extra_kwarg(v)
        return GenerationConfig(**arg_dict, extra_kwargs=extra_kwargs)

    # expose kwargs for OpenAI client (drop `n`)
    def to_kwargs(self, model_meta: ModelMeta) -> dict[str, Any]:
        param_dict = {
            k: v
            for k, v in asdict(self).items()
            if self.__dataclass_fields__[k].metadata.get("forward", True)
        }
        extra_kwargs = param_dict.pop("extra_kwargs", None) or {}
        # apply renaming
        param_renaming = model_meta.param_renaming or {}
        for default_name, custom_name in param_renaming.items():
            if default_name in param_dict:
                param_dict[custom_name] = param_dict.pop(default_name)
            if default_name in extra_kwargs:
                extra_kwargs[custom_name] = extra_kwargs.pop(default_name)
        # apply extra kwargs
        param_dict.update(extra_kwargs)
        # remove unsupported args
        for unsupported_arg in model_meta.unsupported_kw:
            param_dict.pop(unsupported_arg, None)
        return param_dict

    def _preprocess_extra_kwarg(self, val: Any) -> Any:
        # ensure non-schema controlled extra_kwargs are json-serializable
        if isinstance(val, DictConfig):
            return OmegaConf.to_container(val, resolve=True)
        return val


@dataclass
class RetryConfig:
    attempts: int = 5
    wait_min: int = 1
    wait_max: int = 120
