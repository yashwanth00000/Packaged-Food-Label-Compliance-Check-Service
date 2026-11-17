"""
Pydantic schemas for vision-based label extraction.
Strict structured output: every field returns {value, confidence, source_text, image_index}.
If a field is not visible on the label, it must return null with confidence 0.0.
"""

from __future__ import annotations

from typing import Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.schemas import NutrientValue, NutritionalInfo

T = TypeVar("T")


class ExtractedField(BaseModel, Generic[T]):
    """A single extracted field with confidence and source grounding."""
    value: T | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_text: str | None = None
    image_index: int | None = None


class ExtractedNutrientItem(BaseModel):
    """A nutrient row extracted from nutrition table."""
    name: str
    per_100g: float | None = None
    per_100ml: float | None = None
    per_serving: float | None = None
    unit: str = "g"


class ExtractedNutritionTable(BaseModel):
    """Nutrition panel extracted from images."""
    serving_size: str | None = None
    energy_kcal_per_100g: float | None = None
    energy_kcal_per_serving: float | None = None
    nutrients: list[ExtractedNutrientItem] = Field(default_factory=list)

    def to_nutritional_info(self) -> NutritionalInfo:
        """Convert extracted nutrition table to standard pipeline NutritionalInfo."""
        nutrient_values = [
            NutrientValue(
                name=n.name,
                per_100g=n.per_100g,
                per_100ml=n.per_100ml,
                per_serving=n.per_serving,
                unit=n.unit,
            )
            for n in self.nutrients
        ]
        return NutritionalInfo(
            serving_size=self.serving_size,
            energy_kcal_per_100g=self.energy_kcal_per_100g,
            energy_kcal_per_serving=self.energy_kcal_per_serving,
            nutrients=nutrient_values,
        )


class ExtractedLabelData(BaseModel):
    """
    Complete structured extraction from label photos.
    CRITICAL: Never infer, guess, or default unseen values. Unseen fields must be null with confidence 0.0.
    """
    product_name: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    product_category: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    ingredients_list: ExtractedField[list[str]] = Field(default_factory=lambda: ExtractedField[list[str]](value=None, confidence=0.0))
    allergen_declaration: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    net_quantity: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    nutritional_info: ExtractedField[ExtractedNutritionTable] = Field(default_factory=lambda: ExtractedField[ExtractedNutritionTable](value=None, confidence=0.0))
    veg_nonveg_mark: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    fssai_logo_present: ExtractedField[bool] = Field(default_factory=lambda: ExtractedField[bool](value=None, confidence=0.0))
    fssai_licence_number: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    manufacturer_details: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    batch_lot: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    manufacturing_date: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    expiry_date: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    best_before: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    storage_instructions: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    country_of_origin: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))
    claims: ExtractedField[list[str]] = Field(default_factory=lambda: ExtractedField[list[str]](value=None, confidence=0.0))
    other_visible_text: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str](value=None, confidence=0.0))


class ExtractionResult(BaseModel):
    """Result of Stage 0 Label Extraction."""
    extraction_id: str = Field(default_factory=lambda: f"ext-{uuid4().hex}")
    fields: ExtractedLabelData = Field(default_factory=ExtractedLabelData)
    image_quality_warnings: list[str] = Field(default_factory=list)
    image_count: int = 0
    processing_time_seconds: float = 0.0
    fallback_to_manual: bool = False
    error_message: str | None = None
    user_confirmed: bool = False

    @property
    def extracted_data(self) -> ExtractedLabelData:
        """Alias for fields for backward/test compatibility."""
        return self.fields
