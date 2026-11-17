"""
Evaluation Dataset — 22 diverse packaged food labels across multiple categories.
Each sample specifies:
- label: LabelSubmission instance
- expected_recommendation: Recommendation (CLEAR_FOR_PRINT, REVISE_AND_RECHECK, REGULATORY_REVIEW)
- expected_findings: list of dicts with field and issue_type
- category: Food product category
- optional constraints: must_not_have, must_have_issue_types
"""

from __future__ import annotations

from typing import Any

from app.models.enums import IssueType, Recommendation
from app.models.schemas import AdditiveInfo, LabelSubmission, NutrientValue, NutritionalInfo

EVAL_SAMPLES: list[dict[str, Any]] = [
    # 1. Compliant Biscuit
    {
        "name": "compliant_biscuit",
        "category": "Bakery - Biscuits",
        "label": LabelSubmission(
            product_name="Wholesome Wheat Biscuits",
            product_category="Bakery - Biscuits",
            ingredients_list=[
                "Whole Wheat Flour (65%)", "Refined Palm Oil", "Sugar",
                "Invert Sugar Syrup", "Wheat Bran (3%)", "Salt",
                "Raising Agents (INS 500(ii), INS 503(ii))",
                "Emulsifier (INS 322(i))", "Dough Conditioner (INS 223)"
            ],
            allergen_declaration="Contains Wheat, Soy, Sulphite. May contain traces of Nuts and Milk.",
            net_quantity="150 g",
            nutritional_info=NutritionalInfo(
                serving_size="30 g (approx. 4 biscuits)",
                energy_kcal_per_100g=460.0,
                energy_kcal_per_serving=138.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=7.5, per_serving=2.25, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=18.0, per_serving=5.4, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=8.0, per_serving=2.4, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, per_serving=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=68.0, per_serving=20.4, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=22.0, per_serving=6.6, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=19.0, per_serving=5.7, unit="g"),
                    NutrientValue(name="Sodium", per_100g=320.0, per_serving=96.0, unit="mg"),
                    NutrientValue(name="Dietary Fibre", per_100g=6.0, per_serving=1.8, unit="g"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10012011000123",
            manufacturer_details="Wholesome Foods Ltd., Plot 45, KIADB Industrial Area, Bengaluru 560066, Karnataka, India",
            batch_lot="WWB-2024-08A",
            manufacturing_date="2024-08-15",
            expiry_date="2025-02-15",
            best_before="6 months from manufacture",
            storage_instructions="Store in a cool, dry place. Keep in an airtight container once opened.",
            country_of_origin="India",
            additives=[
                AdditiveInfo(name="Sodium Bicarbonate", ins_number="INS 500(ii)", purpose="Raising Agent"),
                AdditiveInfo(name="Ammonium Bicarbonate", ins_number="INS 503(ii)", purpose="Raising Agent"),
                AdditiveInfo(name="Soy Lecithin", ins_number="INS 322(i)", purpose="Emulsifier"),
                AdditiveInfo(name="Sodium Metabisulphite", ins_number="INS 223", purpose="Dough Conditioner"),
            ],
            claims=["Source of fibre"],
            free_text_content="Best enjoyed with tea or coffee.",
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.CLEAR_FOR_PRINT,
        "expected_findings": [],
        "must_not_have": [Recommendation.REGULATORY_REVIEW, Recommendation.REVISE_AND_RECHECK],
    },

    # 2. Missing Mandatory Declarations
    {
        "name": "missing_declarations",
        "category": "Health Supplements - Bars",
        "label": LabelSubmission(
            product_name="Protein Power Bar",
            product_category="Health Supplements - Bars",
            ingredients_list=[
                "Whey Protein Concentrate", "Oats", "Honey", "Almonds",
                "Dark Chocolate Chips", "Soy Lecithin", "Natural Flavour"
            ],
            allergen_declaration=None,  # Missing!
            net_quantity=None,  # Missing!
            nutritional_info=NutritionalInfo(
                serving_size="50 g (1 bar)",
                energy_kcal_per_100g=420.0,
                energy_kcal_per_serving=210.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=25.0, per_serving=12.5, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=14.0, per_serving=7.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=45.0, per_serving=22.5, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=18.0, per_serving=9.0, unit="g"),
                ],
            ),
            veg_nonveg_mark=None,  # Missing!
            fssai_licence_number="10098765432100",
            manufacturer_details=None,  # Missing!
            batch_lot="BX-2024-112",
            manufacturing_date="2024-07-01",
            expiry_date=None,  # Missing!
            storage_instructions=None,  # Missing!
            claims=["High protein", "No added sugar"],
            free_text_content="Fuel your workout!",
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "net_quantity", "issue_type": IssueType.MISSING_DECLARATION.value},
            {"field": "veg_nonveg_mark", "issue_type": IssueType.MISSING_DECLARATION.value},
            {"field": "manufacturer_details", "issue_type": IssueType.MISSING_DECLARATION.value},
            {"field": "expiry_date", "issue_type": IssueType.MISSING_DECLARATION.value},
        ],
        "must_have_issue_types": ["missing_declaration"],
    },

    # 3. Prohibited Health/Medical Claims
    {
        "name": "prohibited_claims",
        "category": "Beverages - Juice",
        "label": LabelSubmission(
            product_name="HerbalVita Immunity Juice",
            product_category="Beverages - Juice",
            ingredients_list=[
                "Water", "Apple Juice Concentrate", "Tulsi Extract",
                "Ashwagandha Extract", "Giloy Extract", "Sugar",
                "Citric Acid (INS 330)", "Permitted Natural Colour (INS 150a)",
                "Sodium Benzoate (INS 211)"
            ],
            allergen_declaration="No known allergens.",
            net_quantity="500 ml",
            nutritional_info=NutritionalInfo(
                serving_size="200 ml",
                energy_kcal_per_100ml=48.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100ml=0.2, unit="g"),
                    NutrientValue(name="Total Fat", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100ml=11.5, unit="g"),
                    NutrientValue(name="Total Sugars", per_100ml=10.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100ml=5.0, unit="g"),
                    NutrientValue(name="Sodium", per_100ml=15.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="20056789012345",
            manufacturer_details="HerbalVita Naturals Inc., 12 Green Park, Delhi 110016, India",
            batch_lot="HV-J-20240901",
            manufacturing_date="2024-09-01",
            expiry_date="2025-03-01",
            best_before="6 months from manufacturing",
            storage_instructions="Refrigerate after opening. Consume within 3 days.",
            country_of_origin="India",
            additives=[
                AdditiveInfo(name="Citric Acid", ins_number="INS 330", purpose="Acidity Regulator"),
                AdditiveInfo(name="Caramel Colour", ins_number="INS 150a", purpose="Colour"),
                AdditiveInfo(name="Sodium Benzoate", ins_number="INS 211", purpose="Preservative"),
            ],
            claims=[
                "Boosts immunity",
                "Cures cold and flu",
                "Immunity booster",
            ],
            free_text_content="Clinically proven to prevent infections. Doctor recommended.",
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REGULATORY_REVIEW,
        "expected_findings": [
            {"field": "claims", "issue_type": IssueType.PROHIBITED_CLAIM.value},
        ],
        "must_have_issue_types": ["prohibited_claim"],
    },

    # 4. Compliant Milk
    {
        "name": "compliant_milk",
        "category": "Dairy - Liquid Milk",
        "label": LabelSubmission(
            product_name="Fresh Dairy Pasteurized Toned Milk",
            product_category="Dairy - Liquid Milk",
            ingredients_list=["Pasteurized Toned Milk", "Vitamin A", "Vitamin D2"],
            allergen_declaration="Contains Milk",
            net_quantity="500 ml",
            nutritional_info=NutritionalInfo(
                serving_size="100 ml",
                energy_kcal_per_100ml=58.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100ml=3.1, unit="g"),
                    NutrientValue(name="Total Fat", per_100ml=3.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100ml=1.9, unit="g"),
                    NutrientValue(name="Trans Fat", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100ml=4.7, unit="g"),
                    NutrientValue(name="Total Sugars", per_100ml=4.7, unit="g"),
                    NutrientValue(name="Added Sugars", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100ml=50.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10014022000555",
            manufacturer_details="Pure Dairy Cooperative, Highway 4, Anand 388001, Gujarat, India",
            batch_lot="MK-20241010-01",
            manufacturing_date="2024-10-10",
            expiry_date="2024-10-12",
            best_before="2 days from packaging",
            storage_instructions="Keep refrigerated at 4°C or below.",
            country_of_origin="India",
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.CLEAR_FOR_PRINT,
        "expected_findings": [],
    },

    # 5. High Sugar Cereal with False Low Sugar Claim
    {
        "name": "high_sugar_cereal",
        "category": "Breakfast Cereals",
        "label": LabelSubmission(
            product_name="Choco Blast Crunchy Flakes",
            product_category="Breakfast Cereals",
            ingredients_list=["Corn (50%)", "Sugar (30%)", "Cocoa Powder", "Barley Malt", "Salt"],
            allergen_declaration="Contains Barley (Gluten).",
            net_quantity="350 g",
            nutritional_info=NutritionalInfo(
                serving_size="30 g",
                energy_kcal_per_100g=390.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=6.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=2.5, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=0.8, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=85.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=32.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=28.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=400.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10015033000444",
            manufacturer_details="Cereal Mills India, Industrial Zone, Pune 411019, India",
            batch_lot="CB-24-901",
            manufacturing_date="2024-09-15",
            expiry_date="2025-06-15",
            storage_instructions="Store in an airtight container.",
            country_of_origin="India",
            claims=["Low sugar"],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "claims", "issue_type": IssueType.UNSUPPORTED_CLAIM.value},
        ],
    },

    # 6. Invalid FSSAI Licence Format
    {
        "name": "invalid_fssai_licence",
        "category": "Snacks - Chips",
        "label": LabelSubmission(
            product_name="Spicy Masala Potato Wafers",
            product_category="Snacks - Chips",
            ingredients_list=["Potatoes", "Palmolein Oil", "Spices and Condiments", "Iodized Salt"],
            allergen_declaration="May contain traces of Peanuts.",
            net_quantity="80 g",
            nutritional_info=NutritionalInfo(
                serving_size="30 g",
                energy_kcal_per_100g=540.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=7.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=34.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=15.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=52.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=2.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=780.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="12345",  # Invalid! Not 14 digits!
            manufacturer_details="Snack Time Foods, Sector 8, Noida 201301, India",
            batch_lot="ST-W-104",
            manufacturing_date="2024-10-01",
            expiry_date="2025-02-01",
            storage_instructions="Store away from direct sunlight.",
            country_of_origin="India",
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "fssai_licence_number", "issue_type": IssueType.FORMATTING_ERROR.value},
        ],
    },

    # 7. Missing Veg/Non-Veg Mark
    {
        "name": "missing_veg_symbol",
        "category": "Dairy - Fats",
        "label": LabelSubmission(
            product_name="Pure Cow Ghee 1L",
            product_category="Dairy - Fats",
            ingredients_list=["Clarified Milk Fat (Cow Milk Ghee)"],
            allergen_declaration="Contains Milk",
            net_quantity="1 L",
            nutritional_info=NutritionalInfo(
                serving_size="15 ml",
                energy_kcal_per_100ml=900.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100ml=99.7, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100ml=64.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100ml=1.5, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100ml=0.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark=None,  # Missing!
            fssai_licence_number="10018044000333",
            manufacturer_details="Ghee Organics Pvt Ltd, Jaipur 302013, Rajasthan, India",
            batch_lot="CG-24-300",
            manufacturing_date="2024-08-01",
            expiry_date="2025-05-01",
            storage_instructions="Store in a cool dry place.",
            country_of_origin="India",
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "veg_nonveg_mark", "issue_type": IssueType.MISSING_DECLARATION.value},
        ],
    },

    # 8. Disease Cure / Prevention Claim
    {
        "name": "disease_cure_claim",
        "category": "Beverages - Herbal Tea",
        "label": LabelSubmission(
            product_name="DiabaShield Herbal Infusion",
            product_category="Beverages - Herbal Tea",
            ingredients_list=["Gymnema Sylvestre", "Fenugreek", "Cinnamon", "Jamun Seed"],
            allergen_declaration="No allergens.",
            net_quantity="100 g (25 tea bags)",
            nutritional_info=NutritionalInfo(
                serving_size="2 g (1 tea bag)",
                energy_kcal_per_100g=120.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=2.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=0.5, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=0.1, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=20.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=0.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=5.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10017055000222",
            manufacturer_details="Ayur Herbals, Dehradun 248001, Uttarakhand, India",
            batch_lot="DH-1004",
            manufacturing_date="2024-09-01",
            expiry_date="2026-09-01",
            storage_instructions="Keep in a cool dry place.",
            country_of_origin="India",
            claims=["Cures diabetes naturally"],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REGULATORY_REVIEW,
        "expected_findings": [
            {"field": "claims", "issue_type": IssueType.PROHIBITED_CLAIM.value},
        ],
    },

    # 9. Compliant Whole Wheat Atta
    {
        "name": "compliant_atta",
        "category": "Food Grains & Flours",
        "label": LabelSubmission(
            product_name="Sharbati Whole Wheat Atta 5kg",
            product_category="Food Grains & Flours",
            ingredients_list=["Whole Wheat"],
            allergen_declaration="Contains Wheat (Gluten).",
            net_quantity="5 kg",
            nutritional_info=NutritionalInfo(
                serving_size="100 g",
                energy_kcal_per_100g=364.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=12.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=1.8, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=0.3, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=72.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=2.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=4.0, unit="mg"),
                    NutrientValue(name="Dietary Fibre", per_100g=11.0, unit="g"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10019066000111",
            manufacturer_details="Agro Milling Corp, Sehore Road, Bhopal 462001, Madhya Pradesh, India",
            batch_lot="ATT-24-88",
            manufacturing_date="2024-10-05",
            expiry_date="2025-01-05",
            best_before="3 months from packaging",
            storage_instructions="Store in clean, cool, and dry container.",
            country_of_origin="India",
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.CLEAR_FOR_PRINT,
        "expected_findings": [],
    },

    # 10. Missing Allergen for Peanuts
    {
        "name": "missing_peanut_allergen",
        "category": "Snacks - Energy Bar",
        "label": LabelSubmission(
            product_name="Nutty Crunch Energy Bar",
            product_category="Snacks - Energy Bar",
            ingredients_list=["Peanuts (40%)", "Oats", "Jaggery", "Peanut Butter", "Chia Seeds"],
            allergen_declaration=None,  # Missing mandatory allergen declaration!
            net_quantity="40 g",
            nutritional_info=NutritionalInfo(
                serving_size="40 g",
                energy_kcal_per_100g=480.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=16.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=24.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=4.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=50.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=22.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=15.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=110.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10020077000999",
            manufacturer_details="Bar Works Ltd, Sector 5, Gurugram 122001, Haryana, India",
            batch_lot="NC-24-12",
            manufacturing_date="2024-09-20",
            expiry_date="2025-03-20",
            storage_instructions="Store in a cool dry place.",
            country_of_origin="India",
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "allergen_declaration", "issue_type": IssueType.MISSING_DECLARATION.value},
        ],
    },

    # 11. Missing Serving Size in Nutrition
    {
        "name": "missing_serving_size",
        "category": "Instant Foods - Noodles",
        "label": LabelSubmission(
            product_name="Spicy Masala Instant Noodles",
            product_category="Instant Foods - Noodles",
            ingredients_list=["Wheat Flour", "Edible Vegetable Oil", "Salt", "Spices (Chilli, Turmeric)"],
            allergen_declaration="Contains Wheat.",
            net_quantity="70 g",
            nutritional_info=NutritionalInfo(
                serving_size=None,  # Missing serving size!
                energy_kcal_per_100g=450.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=9.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=18.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=8.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=63.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=2.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=1150.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10021088000888",
            manufacturer_details="Quick Bite Foods, Peenya, Bengaluru 560058, Karnataka, India",
            batch_lot="MN-24-019",
            manufacturing_date="2024-08-10",
            expiry_date="2025-05-10",
            storage_instructions="Store in cool dry place.",
            country_of_origin="India",
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "nutritional_info", "issue_type": IssueType.MISSING_DECLARATION.value},
        ],
    },

    # 12. Compliant Packaged Natural Mineral Water
    {
        "name": "compliant_mineral_water",
        "category": "Packaged Water",
        "label": LabelSubmission(
            product_name="Himalayan Springs Natural Mineral Water",
            product_category="Packaged Water",
            ingredients_list=["Natural Mineral Water"],
            allergen_declaration="No allergens.",
            net_quantity="1 L",
            nutritional_info=NutritionalInfo(
                serving_size="100 ml",
                energy_kcal_per_100ml=0.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100ml=0.8, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10022099000777",
            manufacturer_details="Himalayan Bottlers, Solan 173212, Himachal Pradesh, India",
            batch_lot="HW-24-77",
            manufacturing_date="2024-10-01",
            expiry_date="2025-04-01",
            best_before="6 months from packaging",
            storage_instructions="Store away from direct sunlight.",
            country_of_origin="India",
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.CLEAR_FOR_PRINT,
        "expected_findings": [],
    },

    # 13. Trans Fat Free Claim Violation
    {
        "name": "trans_fat_claim_violation",
        "category": "Snacks - Savouries",
        "label": LabelSubmission(
            product_name="Crispy Namkeen Mixture",
            product_category="Snacks - Savouries",
            ingredients_list=["Gram Flour", "Refined Vegetable Oil", "Peanuts", "Spices", "Salt"],
            allergen_declaration="Contains Peanuts.",
            net_quantity="200 g",
            nutritional_info=NutritionalInfo(
                serving_size="30 g",
                energy_kcal_per_100g=510.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=11.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=32.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=12.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=1.5, unit="g"),  # Exceeds 0.2g limit for claim!
                    NutrientValue(name="Carbohydrate", per_100g=45.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=2.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=850.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10023011000666",
            manufacturer_details="Bikaner Sweets & Namkeen, Bikaner 334001, Rajasthan, India",
            batch_lot="NM-24-10",
            manufacturing_date="2024-09-01",
            expiry_date="2025-01-01",
            storage_instructions="Store in a cool dry place.",
            country_of_origin="India",
            claims=["Fat free"],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "claims", "issue_type": IssueType.UNSUPPORTED_CLAIM.value},
        ],
    },

    # 14. Missing Net Quantity and Batch Lot
    {
        "name": "missing_quantity_and_batch",
        "category": "Spices & Condiments",
        "label": LabelSubmission(
            product_name="Organic Turmeric Powder",
            product_category="Spices & Condiments",
            ingredients_list=["Turmeric Rhizome Powder"],
            allergen_declaration="No allergens.",
            net_quantity=None,  # Missing!
            nutritional_info=NutritionalInfo(
                serving_size="5 g",
                energy_kcal_per_100g=350.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=8.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=3.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=1.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=65.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=3.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=38.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10024022000555",
            manufacturer_details="Organic Farm Produce Ltd, Erode 638001, Tamil Nadu, India",
            batch_lot=None,  # Missing!
            manufacturing_date="2024-09-10",
            expiry_date="2026-03-10",
            storage_instructions="Store in an airtight container.",
            country_of_origin="India",
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "net_quantity", "issue_type": IssueType.MISSING_DECLARATION.value},
            {"field": "batch_lot", "issue_type": IssueType.MISSING_DECLARATION.value},
        ],
    },

    # 15. Infant Formula Prohibited Claims (IMS Act / FSSAI)
    {
        "name": "infant_formula_prohibited_claims",
        "category": "Infant Nutrition",
        "label": LabelSubmission(
            product_name="BabyGrow Stage 1 Infant Formula",
            product_category="Infant Nutrition",
            ingredients_list=["Demineralized Whey", "Skimmed Milk", "Vegetable Oils", "Lactose", "Minerals", "Vitamins"],
            allergen_declaration="Contains Milk, Soy.",
            net_quantity="400 g",
            nutritional_info=NutritionalInfo(
                serving_size="100 g",
                energy_kcal_per_100g=500.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=10.5, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=26.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=11.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=56.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=50.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=180.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10025033000444",
            manufacturer_details="Infant Health Care Ltd, Sector 18, Gurugram 122015, India",
            batch_lot="IF-24-001",
            manufacturing_date="2024-10-01",
            expiry_date="2025-10-01",
            storage_instructions="Store in a cool dry place.",
            country_of_origin="India",
            claims=["Cures colic and reflux"],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REGULATORY_REVIEW,
        "expected_findings": [
            {"field": "claims", "issue_type": IssueType.PROHIBITED_CLAIM.value},
        ],
    },

    # 16. Compliant Dark Chocolate
    {
        "name": "compliant_dark_chocolate",
        "category": "Confectionery - Chocolate",
        "label": LabelSubmission(
            product_name="Artisan 70% Dark Chocolate Bar",
            product_category="Confectionery - Chocolate",
            ingredients_list=["Cocoa Solids (70%)", "Sugar", "Cocoa Butter", "Emulsifier (INS 322(i))", "Natural Vanilla"],
            allergen_declaration="Contains Soy. May contain Milk and Tree Nuts.",
            net_quantity="80 g",
            nutritional_info=NutritionalInfo(
                serving_size="20 g",
                energy_kcal_per_100g=560.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=8.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=42.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=25.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=38.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=28.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=28.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=20.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10026044000333",
            manufacturer_details="Artisan Chocolatiers, Indiranagar, Bengaluru 560038, Karnataka, India",
            batch_lot="DC-24-05",
            manufacturing_date="2024-09-05",
            expiry_date="2025-09-05",
            best_before="12 months from manufacture",
            storage_instructions="Store in a cool, dry place between 15°C - 20°C.",
            country_of_origin="India",
            additives=[
                AdditiveInfo(name="Soy Lecithin", ins_number="INS 322(i)", purpose="Emulsifier"),
            ],
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.CLEAR_FOR_PRINT,
        "expected_findings": [],
    },

    # 17. Non-Veg Product Missing Non-Veg Mark
    {
        "name": "non_veg_missing_mark",
        "category": "Meat & Poultry Products",
        "label": LabelSubmission(
            product_name="Smoked Chicken Frankfurters",
            product_category="Meat & Poultry Products",
            ingredients_list=["Chicken Meat (75%)", "Water", "Isolated Soy Protein", "Spices", "Iodized Salt", "Preservative (INS 250)"],
            allergen_declaration="Contains Soy.",
            net_quantity="250 g",
            nutritional_info=NutritionalInfo(
                serving_size="50 g",
                energy_kcal_per_100g=210.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=16.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=14.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=5.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=3.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=0.5, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=680.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark=None,  # Missing non-veg brown symbol!
            fssai_licence_number="10027055000222",
            manufacturer_details="Prime Meats Ltd, Industrial Area, Thane 400604, Maharashtra, India",
            batch_lot="CK-24-890",
            manufacturing_date="2024-10-15",
            expiry_date="2024-11-15",
            storage_instructions="Keep frozen at -18°C or below.",
            country_of_origin="India",
            additives=[
                AdditiveInfo(name="Sodium Nitrite", ins_number="INS 250", purpose="Preservative"),
            ],
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "veg_nonveg_mark", "issue_type": IssueType.MISSING_DECLARATION.value},
        ],
    },

    # 18. Compliant Roasted Makhana
    {
        "name": "compliant_roasted_makhana",
        "category": "Healthy Snacks",
        "label": LabelSubmission(
            product_name="Roasted Salt & Pepper Foxnuts (Makhana)",
            product_category="Healthy Snacks",
            ingredients_list=["Foxnuts (Makhana 85%)", "Olive Oil", "Black Pepper", "Himalayan Rock Salt"],
            allergen_declaration="Processed in a facility that also handles Tree Nuts.",
            net_quantity="60 g",
            nutritional_info=NutritionalInfo(
                serving_size="30 g",
                energy_kcal_per_100g=410.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=9.5, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=12.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=2.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=66.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=0.5, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, unit="g"),
                    NutrientValue(name="Dietary Fibre", per_100g=7.5, unit="g"),
                    NutrientValue(name="Sodium", per_100g=450.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10028066000111",
            manufacturer_details="Superfoods India, Sector 62, Noida 201309, Uttar Pradesh, India",
            batch_lot="MK-24-102",
            manufacturing_date="2024-09-25",
            expiry_date="2025-03-25",
            best_before="6 months from manufacture",
            storage_instructions="Store in an airtight container once opened.",
            country_of_origin="India",
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.CLEAR_FOR_PRINT,
        "expected_findings": [],
    },

    # 19. False "No Added Sugar" Claim
    {
        "name": "false_sugar_claim",
        "category": "Beverages - Fruit Drink",
        "label": LabelSubmission(
            product_name="Pure Orchard Apple Delight",
            product_category="Beverages - Fruit Drink",
            ingredients_list=["Water", "Reconstituted Apple Juice (20%)", "Sugar", "Acidity Regulator (INS 330)"],
            allergen_declaration="No allergens.",
            net_quantity="1 L",
            nutritional_info=NutritionalInfo(
                serving_size="200 ml",
                energy_kcal_per_100ml=52.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100ml=0.1, unit="g"),
                    NutrientValue(name="Total Fat", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100ml=13.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100ml=12.5, unit="g"),
                    NutrientValue(name="Added Sugars", per_100ml=8.0, unit="g"),
                    NutrientValue(name="Sodium", per_100ml=15.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10029077000999",
            manufacturer_details="Juice Fresh Ltd, Baramati 413133, Maharashtra, India",
            batch_lot="JD-24-55",
            manufacturing_date="2024-10-01",
            expiry_date="2025-04-01",
            storage_instructions="Refrigerate after opening.",
            country_of_origin="India",
            claims=["No added sugar"],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "claims", "issue_type": IssueType.UNSUPPORTED_CLAIM.value},
        ],
    },

    # 20. Missing Manufacturer Details
    {
        "name": "missing_manufacturer",
        "category": "Edible Oils",
        "label": LabelSubmission(
            product_name="Golden Harvest Refined Sunflower Oil",
            product_category="Edible Oils",
            ingredients_list=["Refined Sunflower Oil", "Vitamin A", "Vitamin D"],
            allergen_declaration="No allergens.",
            net_quantity="1 L",
            nutritional_info=NutritionalInfo(
                serving_size="10 ml",
                energy_kcal_per_100ml=900.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100ml=100.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100ml=11.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100ml=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100ml=0.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10030088000888",
            manufacturer_details=None,  # Missing!
            batch_lot="SO-24-301",
            manufacturing_date="2024-09-01",
            expiry_date="2025-06-01",
            storage_instructions="Store in cool dry place.",
            country_of_origin="India",
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [
            {"field": "manufacturer_details", "issue_type": IssueType.MISSING_DECLARATION.value},
        ],
    },

    # 21. Compliant Mango Pickle
    {
        "name": "compliant_mango_pickle",
        "category": "Pickles & Chutneys",
        "label": LabelSubmission(
            product_name="Traditional Spicy Cut Mango Pickle",
            product_category="Pickles & Chutneys",
            ingredients_list=[
                "Mango Pieces (60%)", "Edible Vegetable Oil (Mustard Oil)", "Iodized Salt",
                "Spices and Condiments (Chilli, Mustard, Fenugreek, Turmeric, Asafoetida)",
                "Acidity Regulator (INS 260)"
            ],
            allergen_declaration="Contains Mustard.",
            net_quantity="400 g",
            nutritional_info=NutritionalInfo(
                serving_size="15 g",
                energy_kcal_per_100g=220.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=2.5, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=16.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=3.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=16.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=2.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=0.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=4200.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10031099000777",
            manufacturer_details="Heritage Foods Co., Guntur 522002, Andhra Pradesh, India",
            batch_lot="MP-24-402",
            manufacturing_date="2024-07-20",
            expiry_date="2025-07-20",
            best_before="12 months from manufacture",
            storage_instructions="Store in a cool dry place. Use a dry spoon.",
            country_of_origin="India",
            additives=[
                AdditiveInfo(name="Acetic Acid", ins_number="INS 260", purpose="Acidity Regulator"),
            ],
            claims=[],
            user_confirmed=True,
        ),
        "expected_recommendation": Recommendation.CLEAR_FOR_PRINT,
        "expected_findings": [],
    },

    # 22. Unconfirmed Photo Extraction (Policy Check)
    {
        "name": "unconfirmed_extraction_rejection",
        "category": "Bakery - Biscuits",
        "label": LabelSubmission(
            product_name="Oatmeal Digestive Biscuits",
            product_category="Bakery - Biscuits",
            ingredients_list=["Rolled Oats (40%)", "Whole Wheat Flour", "Vegetable Oil", "Sugar", "Salt"],
            allergen_declaration="Contains Wheat, Oats.",
            net_quantity="100 g",
            nutritional_info=NutritionalInfo(
                serving_size="25 g",
                energy_kcal_per_100g=450.0,
                nutrients=[
                    NutrientValue(name="Protein", per_100g=8.0, unit="g"),
                    NutrientValue(name="Total Fat", per_100g=17.0, unit="g"),
                    NutrientValue(name="Saturated Fat", per_100g=7.0, unit="g"),
                    NutrientValue(name="Trans Fat", per_100g=0.0, unit="g"),
                    NutrientValue(name="Carbohydrate", per_100g=65.0, unit="g"),
                    NutrientValue(name="Total Sugars", per_100g=16.0, unit="g"),
                    NutrientValue(name="Added Sugars", per_100g=12.0, unit="g"),
                    NutrientValue(name="Sodium", per_100g=300.0, unit="mg"),
                ],
            ),
            veg_nonveg_mark="veg",
            fssai_licence_number="10032011000666",
            manufacturer_details="Healthy Bakes Ltd, Mumbai 400078, Maharashtra, India",
            batch_lot="OD-24-01",
            manufacturing_date="2024-10-01",
            expiry_date="2025-04-01",
            storage_instructions="Store in airtight container.",
            country_of_origin="India",
            claims=[],
            extraction_id="ext-test-1234",
            user_confirmed=False,  # Unconfirmed extraction cannot be CLEAR_FOR_PRINT!
        ),
        "expected_recommendation": Recommendation.REVISE_AND_RECHECK,
        "expected_findings": [],
    },
]
