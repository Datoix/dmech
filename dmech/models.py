from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelBase:
    title: str
    view_radius: float
    trail_length: int = 150
    fps: int = 60
    y_top: float = 0.5
    figsize: Tuple[float, float] = (6, 6)


@dataclass
class RodModel(ModelBase):
    rod_lengths: tuple[float, ...] = ()
    gravity: float = 9.81
    bob_colors: tuple[str, ...] = ("#534ab7", "#0f6e56")
    rod_color: str = "#888480"

    @classmethod
    def from_config(cls, config) -> "RodModel":
        lengths = (config.rod_length,) if hasattr(config, "rod_length") else config.rod_lengths
        return cls(
            title=config.title,
            view_radius=config.view_radius,
            rod_lengths=lengths,
            gravity=config.gravity,
        )


@dataclass
class SpringModel(ModelBase):
    spring_k: float = 40.0
    rest_length: float = 1.0
    gravity: float = 9.81
    bob_colors: tuple[str, ...] = ("#3a86ff", "#ff6b35")
    y_top: float = 0.6
    figsize: Tuple[float, float] = (6, 7)

    @classmethod
    def from_config(cls, config) -> "SpringModel":
        return cls(
            title=f"Spring Double Pendulum via KKT  (k={config.spring_k} N/m, L₀={config.rest_length} m)",
            view_radius=config.view_radius,
            spring_k=config.spring_k,
            rest_length=config.rest_length,
            gravity=config.gravity,
        )


@dataclass
class RackSpec:
    y: float
    length: float
    height: float
    x_index: int


@dataclass
class GearModel(ModelBase):
    gear_centers: tuple[tuple[float, float], ...] = ()
    gear_radii: tuple[float, ...] = ()
    angle_indices: tuple[int, ...] = ()
    gear_colors: tuple[str, ...] = ("#534ab7", "#0f6e56", "#c45c26")
    y_top: float = 1.2
    figsize: Tuple[float, float] = (7, 5)

    @classmethod
    def from_config(cls, config) -> "GearModel":
        return cls(
            title=config.title,
            view_radius=config.view_radius,
            gear_centers=config.gear_centers,
            gear_radii=config.gear_radii,
            angle_indices=config.angle_indices,
        )


@dataclass
class RackPinionModel(ModelBase):
    pinion_center: tuple[float, float] = (0.0, 0.0)
    pinion_radius: float = 0.5
    angle_index: int = 0
    rack: RackSpec | None = None
    pinion_color: str = "#534ab7"
    rack_color: str = "#888480"
    y_top: float = 1.2
    figsize: Tuple[float, float] = (7, 5)

    @classmethod
    def from_config(cls, config) -> "RackPinionModel":
        return cls(
            title=config.title,
            view_radius=config.view_radius,
            pinion_center=config.pinion_center,
            pinion_radius=config.pinion_radius,
            angle_index=config.angle_index,
            rack=config.rack,
        )
