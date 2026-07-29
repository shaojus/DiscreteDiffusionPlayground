import inspect

from playground.data.checkerboard import CheckerboardBinaryStream
from playground.data.gmm_binary_stream import GMMBinaryStream


_MISSING = object()


def _config_get(config, name, default=_MISSING):
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def build_binary_2d_dataset(data_config, device="cpu", realized_state=None):
    dataset = _config_get(data_config, "dataset", None)
    if dataset is None:
        target = str(_config_get(data_config, "_target_", ""))
        if target.endswith(".CheckerboardBinaryStream"):
            dataset = "checkerboard"
        elif target.endswith(".GMMBinaryStream"):
            dataset = "gmm"

    if dataset == "checkerboard":
        dataset_class = CheckerboardBinaryStream
    elif dataset == "gmm":
        dataset_class = GMMBinaryStream
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    parameters = inspect.signature(dataset_class.__init__).parameters
    kwargs = {}
    for name in parameters:
        if name == "self":
            continue
        if name == "device":
            kwargs[name] = str(device)
            continue
        if name == "realized_state":
            kwargs[name] = realized_state
            continue
        value = _config_get(data_config, name)
        if value is not _MISSING:
            kwargs[name] = value
    return dataset_class(**kwargs)
