from dataclasses import dataclass


@dataclass
class LoRARequest:
    """
        请求一个 LoRA 适配器。

        请注意，此类应仅在内部使用。对于在线服务，建议不要允许用户直接使用此类，而是提供另一层抽象以防止用户访问未经授权的 LoRA 适配器。

        lora_int_id 对于给定的适配器必须全局唯一
    """

    lora_name: str
    lora_int_id: int
    lora_local_path: str

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(
                f"lora_int_id must be > 0, got {self.lora_int_id}")

    def __eq__(self, value: object) -> bool:
        return isinstance(
            value, LoRARequest) and self.lora_int_id == value.lora_int_id

    def __hash__(self) -> int:
        return self.lora_int_id
