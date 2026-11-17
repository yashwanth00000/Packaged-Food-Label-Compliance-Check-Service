"""
Held-out Evaluation Dataset for FSSAI Label Compliance.

CRITICAL NOTE:
This dataset is STRICTLY HELD-OUT and was NEVER used during development to tune
pipeline heuristics, prompt templates, or rule thresholds.
The 22 samples in `eval/samples.py` constitute the DEVELOPMENT / TUNING dataset.

Each sample here represents an independent packaged food product across diverse categories:
- Grains & Millets
- Dairy & Alternatives
- Condiments & Oils
- Snacks & Ready-to-Cook
- Beverages
"""

from __future__ import annotations

from typing import Any

from app.models.enums import Recommendation
from app.models.schemas import LabelSubmission, NutrientValue, NutritionalInfo

HELDOUT_SAMPLES: list[dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # Sample H01: Compliant Ragi Dosa Instant Breakfast Mix (CLEAR_FOR_PRINT)
    # -------------------------------------------------------------------------
    {
        "id": "H01",
        "name": "Ragi Dosa Instant Breakfast Mix",
        "category": "Grains & Millets",
        "expected_recommendation": Recommendation.CLEAR_FOR_PRINT,
        "expected_findings": [],
        "label": LabelSubmission(
            product_name="Millet Magic Instant Ragi Dosa Mix",
            product_category="Ready-to-Cook Cereal Mixes",
            ingredients_list=["Finger Millet (Ragi) Flour (45%)", "Rice Flour (30%)", "Black Gram (Urad) Dal (20%)", "Salt", "Fenugreek Powder"],
            allergen_declaration="May contain traces of gluten and soy.",
            net_quantity="500 g",
            nutritional_info=NutritionalInfo(
                serving_size="50g",
                energy_kcal_per_100g=340.0,
                energy_kcal_per_serving=170.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=9.8, per_serving=4.9, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=72.0, per_serving=36.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=1.2, per_serving=0.6, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=1.8, per_serving=0.9, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=0.4, per_serving=0.2, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=450.0, per_serving=225.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10012011000456",
            manufacturer_details="Deccan Millet Foods Pvt Ltd, Plot 12, Industrial Area, Hubli, Karnataka 580020",
            batch_lot="LOT-RD-2024-88",
            manufacturing_date="2024-06-01",
            expiry_date="2024-12-01",
            storage_instructions="Store in a cool dry hygienic place away from direct sunlight.",
            country_of_origin="India",
            user_confirmed=True,
        ),
    },

    # -------------------------------------------------------------------------
    # Sample H02: Pure Malai Paneer (CLEAR_FOR_PRINT)
    # -------------------------------------------------------------------------
    {
        "id": "H02",
        "name": "Fresh Malai Paneer 200g",
        "category": "Dairy Products",
        "expected_recommendation": Recommendation.CLEAR_FOR_PRINT,
        "expected_findings": [],
        "label": LabelSubmission(
            product_name="Gokul Pure Malai Paneer",
            product_category="Dairy",
            ingredients_list=["Pasteurised Whole Cow Milk", "Citric Acid (Coagulant)"],
            allergen_declaration="Contains Milk.",
            net_quantity="200 g",
            nutritional_info=NutritionalInfo(
                serving_size="100g",
                energy_kcal_per_100g=295.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=18.5, per_serving=18.5, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=3.2, per_serving=3.2, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=2.8, per_serving=2.8, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=23.0, per_serving=23.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=14.5, per_serving=14.5, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.5, per_serving=0.5, unit="g"),
                    NutrientValue(name="Sodium", per_100g=30.0, per_serving=30.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="11518018000234",
            manufacturer_details="Kolhapur Zilla Sahakari Dudh Utpadak Sangh Ltd, B-1 MIDC Gokul Shirgaon, Kolhapur 416234",
            batch_lot="PNR-109",
            manufacturing_date="2024-06-15",
            expiry_date="2024-07-05",
            storage_instructions="Keep refrigerated at 4 deg C or below.",
            country_of_origin="India",
            user_confirmed=True,
        ),
    },

    # -------------------------------------------------------------------------
    # Sample H03: Traditional Basmati Rice 5kg (CLEAR_FOR_PRINT)
    # -------------------------------------------------------------------------
    {
        "id": "H03",
        "name": "Traditional Royal Basmati Rice 5kg",
        "category": "Grains & Cereals",
        "expected_recommendation": Recommendation.CLEAR_FOR_PRINT,
        "expected_findings": [],
        "label": LabelSubmission(
            product_name="Heritage Royal Aged Basmati Rice",
            product_category="Raw Agricultural Grains",
            ingredients_list=["Aged Basmati Rice Grain"],
            allergen_declaration="Contains no major allergens.",
            net_quantity="5 kg",
            nutritional_info=NutritionalInfo(
                serving_size="100g",
                energy_kcal_per_100g=350.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=8.5, per_serving=8.5, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=77.0, per_serving=77.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=0.5, per_serving=0.5, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=0.1, per_serving=0.1, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=2.0, per_serving=2.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10014064000333",
            manufacturer_details="Punjab Grain Exporters Pvt Ltd, GT Road, Karnal, Haryana 132001",
            batch_lot="BMT-2024-09",
            manufacturing_date="2024-05-10",
            best_before="24 months from packaging",
            storage_instructions="Store in a cool and dry container.",
            country_of_origin="India",
            user_confirmed=True,
        ),
    },

    # -------------------------------------------------------------------------
    # Sample H04: Canned Coconut Milk missing Batch Lot (REVISE_AND_RECHECK)
    # -------------------------------------------------------------------------
    {
        "id": "H04",
        "name": "Canned Coconut Milk Missing Batch Lot",
        "category": "Canned Foods",
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "batch_lot", "issue_type": "missing_declaration"},
        ],
        "label": LabelSubmission(
            product_name="Kerala Gold Organic Coconut Milk",
            product_category="Processed Coconut Products",
            ingredients_list=["Coconut Extract (70%)", "Water", "Guar Gum (INS 412)"],
            allergen_declaration="",
            net_quantity="400 ml",
            nutritional_info=NutritionalInfo(
                serving_size="100ml",
                energy_kcal_per_100ml=180.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=1.8, per_serving=1.8, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=2.5, per_serving=2.5, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=1.5, per_serving=1.5, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=18.0, per_serving=18.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=16.0, per_serving=16.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=15.0, per_serving=15.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="11317007000199",
            manufacturer_details="Malabar Agro Processing Co, Kozhikode, Kerala 673001",
            batch_lot=None,  # Missing mandatory declaration
            manufacturing_date="2024-04-12",
            expiry_date="2025-04-11",
            storage_instructions="Shake well before use. Refrigerate after opening.",
            country_of_origin="India",
            user_confirmed=True,
        ),
    },

    # -------------------------------------------------------------------------
    # Sample H05: Peanut Butter with Peanut Allergen Omitted (REGULATORY_REVIEW)
    # -------------------------------------------------------------------------
    {
        "id": "H05",
        "name": "Peanut Butter with Missing Allergen Statement",
        "category": "Spreads & Nut Butters",
        "expected_recommendation": Recommendation.REGULATORY_REVIEW,
        "expected_findings": [
            {"field": "allergen_declaration", "issue_type": "allergen_mismatch"},
        ],
        "label": LabelSubmission(
            product_name="FitPro Roasted Peanut Butter Crunchy",
            product_category="Nut Butters",
            ingredients_list=["Roasted Peanuts (92%)", "Brown Sugar", "Hydrogenated Soybean Oil", "Iodised Salt"],
            allergen_declaration="Contains no common allergens.",  # Contradicts peanut & soy ingredients!
            net_quantity="350 g",
            nutritional_info=NutritionalInfo(
                serving_size="32g",
                energy_kcal_per_100g=620.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=26.0, per_serving=8.3, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=19.0, per_serving=6.1, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=6.0, per_serving=1.9, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=3.0, per_serving=1.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=50.0, per_serving=16.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=10.0, per_serving=3.2, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=180.0, per_serving=57.6, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10719024000122",
            manufacturer_details="Gujarat Nut Foods LLP, Rajkot Gondal Highway, Gujarat 360311",
            batch_lot="PB-994",
            manufacturing_date="2024-05-18",
            expiry_date="2025-05-17",
            storage_instructions="Store in a cool dry place. Oil separation is natural.",
            country_of_origin="India",
            user_confirmed=True,
        ),
    },

    # -------------------------------------------------------------------------
    # Sample H06: Packaged Fruit Drink with Cancer Prevention Claim (REGULATORY_REVIEW)
    # -------------------------------------------------------------------------
    {
        "id": "H06",
        "name": "Fruit Drink with Prohibited Disease Prevention Claim",
        "category": "Beverages",
        "expected_recommendation": Recommendation.REGULATORY_REVIEW,
        "expected_findings": [
            {"field": "claims", "issue_type": "prohibited_claim"},
        ],
        "label": LabelSubmission(
            product_name="Pomegranate Miracle Elixir",
            product_category="Fruit Juice Based Beverage",
            ingredients_list=["Water", "Pomegranate Juice Concentrate (18%)", "Sugar", "Acidity Regulator (INS 330)"],
            allergen_declaration="",
            net_quantity="250 ml",
            nutritional_info=NutritionalInfo(
                serving_size="250ml",
                energy_kcal_per_100ml=52.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=0.2, per_serving=0.5, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=12.5, per_serving=31.2, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=11.8, per_serving=29.5, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=8.0, per_serving=20.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=12.0, per_serving=30.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10018022007890",
            manufacturer_details="Apex Beverages Ltd, Hinjewadi Phase 3, Pune 411057",
            batch_lot="POM-12",
            manufacturing_date="2024-06-10",
            best_before="6 months from manufacture",
            storage_instructions="Store away from sunlight. Consume immediately after opening.",
            country_of_origin="India",
            claims=["Prevents cancer and treats heart disease"],  # Strictly prohibited under Clause 4
            user_confirmed=True,
        ),
    },

    # -------------------------------------------------------------------------
    # Sample H07: Mustard Oil with Argemone-Free Declaration (CLEAR_FOR_PRINT)
    # -------------------------------------------------------------------------
    {
        "id": "H07",
        "name": "Cold Pressed Kachi Ghani Mustard Oil 1L",
        "category": "Edible Oils",
        "expected_recommendation": Recommendation.CLEAR_FOR_PRINT,
        "expected_findings": [],
        "label": LabelSubmission(
            product_name="Shree Kachi Ghani Pure Mustard Oil",
            product_category="Edible Vegetable Oil",
            ingredients_list=["100% Pure Mustard Seed Oil"],
            allergen_declaration="Contains Mustard.",
            net_quantity="1 L",
            nutritional_info=NutritionalInfo(
                serving_size="15g",
                energy_kcal_per_100g=900.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=100.0, per_serving=15.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=7.5, per_serving=1.1, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=0.0, per_serving=0.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="12216027000145",
            manufacturer_details="Rajasthan Oil Mills Ltd, Alwar Industrial Area, Alwar 301030",
            batch_lot="MO-2024-K1",
            manufacturing_date="2024-05-25",
            best_before="9 months from packaging",
            storage_instructions="Store in a cool dry place protected from sunlight.",
            country_of_origin="India",
            user_confirmed=True,
        ),
    },

    # -------------------------------------------------------------------------
    # Sample H08: Chicken Masala Marked Vegetarian (REGULATORY_REVIEW)
    # -------------------------------------------------------------------------
    {
        "id": "H08",
        "name": "Chicken Gravy Paste Marked Vegetarian",
        "category": "Condiments & Sauces",
        "expected_recommendation": Recommendation.REGULATORY_REVIEW,
        "expected_findings": [
            {"field": "veg_nonveg_mark", "issue_type": "inconsistency"},
        ],
        "label": LabelSubmission(
            product_name="Dhaba Style Rich Chicken Curry Base",
            product_category="Culinary Pastes",
            ingredients_list=["Onion Paste", "Tomato Puree", "Chicken Fat", "Ginger Garlic Paste", "Spices", "Refined Sunflower Oil"],
            allergen_declaration="",
            net_quantity="200 g",
            nutritional_info=NutritionalInfo(
                serving_size="50g",
                energy_kcal_per_100g=220.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=4.2, per_serving=2.1, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=12.0, per_serving=6.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=4.5, per_serving=2.2, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=1.0, per_serving=0.5, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=17.5, per_serving=8.75, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=5.0, per_serving=2.5, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=880.0, per_serving=440.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",  # FATAL ERROR: Chicken fat ingredient marked as vegetarian!
            fssai_licence_number="10020051000888",
            manufacturer_details="Taste of Punjab Foods, Phase 7 Mohali, Punjab 160055",
            batch_lot="CH-882",
            manufacturing_date="2024-06-02",
            expiry_date="2024-12-01",
            storage_instructions="Store in refrigerator after opening.",
            country_of_origin="India",
            user_confirmed=True,
        ),
    },
]
