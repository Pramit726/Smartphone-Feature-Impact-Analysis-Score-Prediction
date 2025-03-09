from typing import List, Optional

from pydantic import BaseModel


class SmartphoneSpecs(BaseModel):
    price: int
    brand_name: str
    has_5g: bool
    has_nfc: bool
    has_ir_blaster: bool
    num_cores: float
    processor_speed: float
    processor_brand: str
    ram_capacity: float
    internal_memory: float
    fast_charging: Optional[float] = None
    screen_size: float
    resolution: str
    refresh_rate: int
    num_rear_cameras: int
    num_front_cameras: int
    primary_camera_rear: str
    primary_camera_front: float
    fast_charging_available: int
    extended_memory_available: int
    extended_upto: Optional[float] = None


class ShowInsights(BaseModel):
    points: List[str]
    info_message: str


class ShowRatings(BaseModel):
    rating: str


class ShowInterpretation(BaseModel):
    summary_plot: Optional[str]
    force_plot: str


class ShowCatagories(BaseModel):
    brand_categories: List[str]
    processor_categories: List[str]
    resolution_categories: List[str]
    primary_camera_rear_categories: List[str]


class ShowModelMetrics(BaseModel):
    mae: float
    r2_score: float
