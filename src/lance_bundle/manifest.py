from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelSignature:
    distance_metric: Optional[str] = None
    max_seq_length: Optional[int] = None

    @classmethod
    def from_json(cls, data: dict) -> "ModelSignature":
        return cls(
            distance_metric=data.get("distance_metric"),
            max_seq_length=data.get("max_seq_length"),
        )

    def to_json(self) -> dict:
        result = {}
        if self.distance_metric is not None:
            result["distance_metric"] = self.distance_metric
        if self.max_seq_length is not None:
            result["max_seq_length"] = self.max_seq_length
        return result

@dataclass
class ModelComponent:
    type: str
    path: str
    generator: str
    model_card: dict
    signature: ModelSignature

    @classmethod
    def from_json(cls, data: dict) -> "ModelComponent":
        return cls(
            type=data["type"],
            path=data["path"],
            generator=data["generator"],
            model_card=data["model_card"],
            signature=ModelSignature.from_json(data.get("signature", {})),
        )

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "path": self.path,
            "generator": self.generator,
            "model_card": self.model_card,
            "signature": self.signature.to_json(),
        }

@dataclass
class DataComponent:
    type: str
    path: str
    dataset_name: Optional[str] = None
    dataset_description: Optional[str] = None
    dataset_source: Optional[str] = None

    @classmethod
    def from_json(cls, data: dict) -> "DataComponent":
        return cls(
            type=data["type"],
            path=data["path"],
            dataset_name=data.get("dataset_name"),
            dataset_description=data.get("dataset_description"),
            dataset_source=data.get("dataset_source"),
        )

    def to_json(self) -> dict:
        result = {
            "type": self.type,
            "path": self.path,
        }
        if self.dataset_name is not None:
            result["dataset_name"] = self.dataset_name
        if self.dataset_description is not None:
            result["dataset_description"] = self.dataset_description
        if self.dataset_source is not None:
            result["dataset_source"] = self.dataset_source
        return result

@dataclass
class Manifest:
    version: str
    generator: str
    created_at: str
    model: ModelComponent
    data: DataComponent

    @classmethod
    def from_json(cls, data: dict) -> "Manifest":
        components = data["components"]
        return cls(
            version=data["version"],
            generator=data["generator"],
            created_at=data["created_at"],
            model=ModelComponent.from_json(components["model"]),
            data=DataComponent.from_json(components["data"]),
        )

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "generator": self.generator,
            "created_at": self.created_at,
            "components": {
                "model": self.model.to_json(),
                "data": self.data.to_json(),
            }
        }
