"""
Generate 10 synthetic test images of packaged food labels for evaluation.
Uses PIL to draw realistic label elements (text, frames, veg/non-veg logos, tables).
Also generates eval/images/ground_truth.json with exact ground-truth values.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 10 Label Definitions with ground-truth fields
TEST_LABELS = [
    {
        "filename": "01_biscuit_label.png",
        "title": "GoodHealth Wheat Biscuits",
        "category": "Bakery - Biscuits",
        "lines": [
            "GoodHealth Wheat Biscuits",
            "Ingredients: Whole Wheat Flour (60%), Sugar, Palm Oil, Salt, Raising Agent (INS 500ii).",
            "Contains: Wheat. May contain Milk.",
            "Net Qty: 120 g",
            "Energy: 450 kcal/100g | Protein: 7g | Total Fat: 15g | Total Sugars: 20g | Sodium: 300mg",
            "Veg Mark: [GREEN CIRCLE]",
            "FSSAI Lic. No.: 10019011000123",
            "Mfg by: HealthBake Foods Pvt Ltd, Bangalore 560001",
            "Batch: GHB-01 | Mfg Date: 2024-08-01 | Best Before: 6 months from mfg",
            "Store in a cool dry place.",
            "Claims: Source of fibre"
        ],
        "is_veg": True,
        "ground_truth": {
            "product_name": "GoodHealth Wheat Biscuits",
            "product_category": "Bakery - Biscuits",
            "ingredients_list": ["Whole Wheat Flour (60%)", "Sugar", "Palm Oil", "Salt", "Raising Agent (INS 500ii)"],
            "allergen_declaration": "Contains: Wheat. May contain Milk.",
            "net_quantity": "120 g",
            "veg_nonveg_mark": "veg",
            "fssai_licence_number": "10019011000123",
            "manufacturer_details": "HealthBake Foods Pvt Ltd, Bangalore 560001",
            "batch_lot": "GHB-01",
            "manufacturing_date": "2024-08-01",
            "expiry_date": None,
            "best_before": "6 months from mfg",
            "storage_instructions": "Store in a cool dry place.",
            "country_of_origin": None,
            "claims": ["Source of fibre"],
        }
    },
    {
        "filename": "02_mango_juice.png",
        "title": "SunRipe Mango Nectar",
        "category": "Beverages - Fruit Drink",
        "lines": [
            "SunRipe Mango Nectar",
            "Ingredients: Water, Mango Pulp (25%), Sugar, Acidity Regulator (INS 330), Antioxidant (INS 300).",
            "Net Qty: 1 L",
            "Energy: 54 kcal/100ml | Total Sugars: 13g | Added Sugars: 9g | Protein: 0.2g | Fat: 0g",
            "Veg Mark: [GREEN CIRCLE]",
            "FSSAI Lic. No.: 10018022000456",
            "Mfg by: SunRipe Beverages Ltd, Pune 411001, Maharashtra",
            "Batch: MN-2401 | Mfg: 2024-09-10 | Expiry: 2025-03-10",
            "Refrigerate after opening.",
            "Claims: Rich in Vitamin C"
        ],
        "is_veg": True,
        "ground_truth": {
            "product_name": "SunRipe Mango Nectar",
            "product_category": "Beverages - Fruit Drink",
            "ingredients_list": ["Water", "Mango Pulp (25%)", "Sugar", "Acidity Regulator (INS 330)", "Antioxidant (INS 300)"],
            "allergen_declaration": None,
            "net_quantity": "1 L",
            "veg_nonveg_mark": "veg",
            "fssai_licence_number": "10018022000456",
            "manufacturer_details": "SunRipe Beverages Ltd, Pune 411001, Maharashtra",
            "batch_lot": "MN-2401",
            "manufacturing_date": "2024-09-10",
            "expiry_date": "2025-03-10",
            "best_before": None,
            "storage_instructions": "Refrigerate after opening.",
            "country_of_origin": None,
            "claims": ["Rich in Vitamin C"],
        }
    },
    {
        "filename": "03_protein_bar_missing.png",
        "title": "Turbo Fuel Protein Bar",
        "category": "Health Supplements - Bars",
        "lines": [
            "Turbo Fuel Protein Bar",
            "Ingredients: Whey Protein, Almonds, Cocoa, Honey.",
            "Energy: 220 kcal/bar | Protein: 20g | Fat: 8g | Carbs: 18g",
            "FSSAI Lic. No.: 10020033000789",
            "Batch: TF-99 | Mfg Date: 2024-07-15",
            "Claims: High protein, No added sugar"
        ],
        "is_veg": None,
        "ground_truth": {
            "product_name": "Turbo Fuel Protein Bar",
            "product_category": "Health Supplements - Bars",
            "ingredients_list": ["Whey Protein", "Almonds", "Cocoa", "Honey"],
            "allergen_declaration": None,
            "net_quantity": None,
            "veg_nonveg_mark": None,
            "fssai_licence_number": "10020033000789",
            "manufacturer_details": None,
            "batch_lot": "TF-99",
            "manufacturing_date": "2024-07-15",
            "expiry_date": None,
            "best_before": None,
            "storage_instructions": None,
            "country_of_origin": None,
            "claims": ["High protein", "No added sugar"],
        }
    },
    {
        "filename": "04_potato_chips.png",
        "title": "Crunchy Classic Salted Chips",
        "category": "Snacks - Chips",
        "lines": [
            "Crunchy Classic Salted Chips",
            "Ingredients: Potato, Palmolein Oil, Iodised Salt.",
            "May contain traces of Peanuts.",
            "Net Qty: 50 g",
            "Energy: 540 kcal/100g | Protein: 6g | Total Fat: 33g | Carbs: 55g | Sodium: 650mg",
            "Veg Mark: [GREEN CIRCLE]",
            "FSSAI Lic. No.: 10021044000321",
            "Mfg by: Crunchy Snacks India, Noida 201301",
            "Batch: CS-101 | Mfg: 2024-10-01 | Best Before: 4 months from mfg",
            "Store in a cool and dry place."
        ],
        "is_veg": True,
        "ground_truth": {
            "product_name": "Crunchy Classic Salted Chips",
            "product_category": "Snacks - Chips",
            "ingredients_list": ["Potato", "Palmolein Oil", "Iodised Salt"],
            "allergen_declaration": "May contain traces of Peanuts.",
            "net_quantity": "50 g",
            "veg_nonveg_mark": "veg",
            "fssai_licence_number": "10021044000321",
            "manufacturer_details": "Crunchy Snacks India, Noida 201301",
            "batch_lot": "CS-101",
            "manufacturing_date": "2024-10-01",
            "expiry_date": None,
            "best_before": "4 months from mfg",
            "storage_instructions": "Store in a cool and dry place.",
            "country_of_origin": None,
            "claims": [],
        }
    },
    {
        "filename": "05_pasteurised_milk.png",
        "title": "DairyFresh Pure Toned Milk",
        "category": "Dairy - Liquid Milk",
        "lines": [
            "DairyFresh Pure Toned Milk",
            "Ingredients: Toned Milk, Vitamin A, Vitamin D2.",
            "Contains: Milk.",
            "Net Qty: 500 ml",
            "Energy: 58 kcal/100ml | Protein: 3.1g | Fat: 3.0g | Saturated Fat: 1.9g | Calcium: 120mg",
            "Veg Mark: [GREEN CIRCLE]",
            "FSSAI Lic. No.: 10014022000555",
            "Mfg by: DairyFresh Dairy Co-op, Anand 388001, Gujarat",
            "Batch: DF-2410 | Use By: 2024-10-25",
            "Keep refrigerated at 4°C or below."
        ],
        "is_veg": True,
        "ground_truth": {
            "product_name": "DairyFresh Pure Toned Milk",
            "product_category": "Dairy - Liquid Milk",
            "ingredients_list": ["Toned Milk", "Vitamin A", "Vitamin D2"],
            "allergen_declaration": "Contains: Milk.",
            "net_quantity": "500 ml",
            "veg_nonveg_mark": "veg",
            "fssai_licence_number": "10014022000555",
            "manufacturer_details": "DairyFresh Dairy Co-op, Anand 388001, Gujarat",
            "batch_lot": "DF-2410",
            "manufacturing_date": None,
            "expiry_date": "2024-10-25",
            "best_before": None,
            "storage_instructions": "Keep refrigerated at 4°C or below.",
            "country_of_origin": None,
            "claims": [],
        }
    },
    {
        "filename": "06_wheat_atta.png",
        "title": "Sharbati Whole Wheat Atta 5kg",
        "category": "Food Grains & Flours",
        "lines": [
            "Sharbati Whole Wheat Atta 5kg",
            "Ingredients: Whole Wheat.",
            "Contains: Wheat (Gluten).",
            "Net Qty: 5 kg",
            "Energy: 364 kcal/100g | Protein: 12g | Dietary Fibre: 11g | Carbohydrate: 72g",
            "Veg Mark: [GREEN CIRCLE]",
            "FSSAI Lic. No.: 10019066000111",
            "Mfg by: Agro Mills India, Bhopal 462001, MP",
            "Batch: SWA-88 | Mfg Date: 2024-09-01 | Best Before: 3 months from packaging",
            "Store in clean, dry airtight container.",
            "Claims: 100% Whole Wheat, Rich in Dietary Fibre"
        ],
        "is_veg": True,
        "ground_truth": {
            "product_name": "Sharbati Whole Wheat Atta 5kg",
            "product_category": "Food Grains & Flours",
            "ingredients_list": ["Whole Wheat"],
            "allergen_declaration": "Contains: Wheat (Gluten).",
            "net_quantity": "5 kg",
            "veg_nonveg_mark": "veg",
            "fssai_licence_number": "10019066000111",
            "manufacturer_details": "Agro Mills India, Bhopal 462001, MP",
            "batch_lot": "SWA-88",
            "manufacturing_date": "2024-09-01",
            "expiry_date": None,
            "best_before": "3 months from packaging",
            "storage_instructions": "Store in clean, dry airtight container.",
            "country_of_origin": None,
            "claims": ["100% Whole Wheat", "Rich in Dietary Fibre"],
        }
    },
    {
        "filename": "07_turmeric_powder.png",
        "title": "Golden Pure Turmeric Powder",
        "category": "Spices & Condiments",
        "lines": [
            "Golden Pure Turmeric Powder",
            "Ingredients: Dried Turmeric Rhizome.",
            "Net Qty: 200 g",
            "Energy: 350 kcal/100g | Curcumin content: 3%",
            "Veg Mark: [GREEN CIRCLE]",
            "FSSAI Lic. No.: 10024022000555",
            "Mfg by: Spice Roots Ltd, Erode 638001, Tamil Nadu",
            "Mfg Date: 2024-08-15 | Best Before: 12 months from mfg",
            "Store in an airtight jar in a cool dry place."
        ],
        "is_veg": True,
        "ground_truth": {
            "product_name": "Golden Pure Turmeric Powder",
            "product_category": "Spices & Condiments",
            "ingredients_list": ["Dried Turmeric Rhizome"],
            "allergen_declaration": None,
            "net_quantity": "200 g",
            "veg_nonveg_mark": "veg",
            "fssai_licence_number": "10024022000555",
            "manufacturer_details": "Spice Roots Ltd, Erode 638001, Tamil Nadu",
            "batch_lot": None,
            "manufacturing_date": "2024-08-15",
            "expiry_date": None,
            "best_before": "12 months from mfg",
            "storage_instructions": "Store in an airtight jar in a cool dry place.",
            "country_of_origin": None,
            "claims": [],
        }
    },
    {
        "filename": "08_packaged_water.png",
        "title": "Himalayan Springs Natural Mineral Water",
        "category": "Packaged Water",
        "lines": [
            "Himalayan Springs Natural Mineral Water",
            "Ingredients: Natural Mineral Water.",
            "Net Qty: 1 L",
            "Minerals: Calcium 3.5mg, Magnesium 1.2mg, Sodium 0.8mg per 100ml",
            "Veg Mark: [GREEN CIRCLE]",
            "FSSAI Lic. No.: 10022099000777",
            "Bottled by: Himalayan Waters Pvt Ltd, Solan 173212, HP",
            "Batch: HW-771 | Mfg: 2024-10-01 | Best Before: 6 months",
            "Keep away from direct sunlight."
        ],
        "is_veg": True,
        "ground_truth": {
            "product_name": "Himalayan Springs Natural Mineral Water",
            "product_category": "Packaged Water",
            "ingredients_list": ["Natural Mineral Water"],
            "allergen_declaration": None,
            "net_quantity": "1 L",
            "veg_nonveg_mark": "veg",
            "fssai_licence_number": "10022099000777",
            "manufacturer_details": "Himalayan Waters Pvt Ltd, Solan 173212, HP",
            "batch_lot": "HW-771",
            "manufacturing_date": "2024-10-01",
            "expiry_date": None,
            "best_before": "6 months",
            "storage_instructions": "Keep away from direct sunlight.",
            "country_of_origin": None,
            "claims": [],
        }
    },
    {
        "filename": "09_dark_chocolate.png",
        "title": "Artisan 70% Dark Chocolate",
        "category": "Confectionery - Chocolate",
        "lines": [
            "Artisan 70% Dark Chocolate",
            "Ingredients: Cocoa Mass, Cocoa Butter, Sugar, Emulsifier (INS 322i), Vanilla.",
            "Contains: Soy. May contain Milk, Tree Nuts.",
            "Net Qty: 80 g",
            "Energy: 560 kcal/100g | Protein: 8g | Total Fat: 42g | Carbs: 38g",
            "Veg Mark: [GREEN CIRCLE]",
            "FSSAI Lic. No.: 10026044000333",
            "Mfg by: Artisan Chocolates Ltd, Bangalore 560038",
            "Batch: DC-05 | Mfg: 2024-09-05 | Expiry: 2025-09-05",
            "Store in a cool, dry place at 15-20°C.",
            "Claims: Rich in Cocoa"
        ],
        "is_veg": True,
        "ground_truth": {
            "product_name": "Artisan 70% Dark Chocolate",
            "product_category": "Confectionery - Chocolate",
            "ingredients_list": ["Cocoa Mass", "Cocoa Butter", "Sugar", "Emulsifier (INS 322i)", "Vanilla"],
            "allergen_declaration": "Contains: Soy. May contain Milk, Tree Nuts.",
            "net_quantity": "80 g",
            "veg_nonveg_mark": "veg",
            "fssai_licence_number": "10026044000333",
            "manufacturer_details": "Artisan Chocolates Ltd, Bangalore 560038",
            "batch_lot": "DC-05",
            "manufacturing_date": "2024-09-05",
            "expiry_date": "2025-09-05",
            "best_before": None,
            "storage_instructions": "Store in a cool, dry place at 15-20°C.",
            "country_of_origin": None,
            "claims": ["Rich in Cocoa"],
        }
    },
    {
        "filename": "10_herbal_tea_claims.png",
        "title": "AyurVida Wellness Herbal Infusion",
        "category": "Beverages - Herbal Tea",
        "lines": [
            "AyurVida Wellness Herbal Infusion",
            "Ingredients: Tulsi, Ginger, Cardamom, Clove, Cinnamon.",
            "Net Qty: 50 g (25 Tea Bags)",
            "Veg Mark: [GREEN CIRCLE]",
            "FSSAI Lic. No.: 10017055000222",
            "Mfg by: AyurVida Natural Products, Dehradun 248001, UK",
            "Batch: AV-88 | Mfg: 2024-09-01 | Best Before: 24 months from mfg",
            "Store in airtight container away from moisture.",
            "Claims: Cures diabetes naturally, Boosts immunity 100%"
        ],
        "is_veg": True,
        "ground_truth": {
            "product_name": "AyurVida Wellness Herbal Infusion",
            "product_category": "Beverages - Herbal Tea",
            "ingredients_list": ["Tulsi", "Ginger", "Cardamom", "Clove", "Cinnamon"],
            "allergen_declaration": None,
            "net_quantity": "50 g (25 Tea Bags)",
            "veg_nonveg_mark": "veg",
            "fssai_licence_number": "10017055000222",
            "manufacturer_details": "AyurVida Natural Products, Dehradun 248001, UK",
            "batch_lot": "AV-88",
            "manufacturing_date": "2024-09-01",
            "expiry_date": None,
            "best_before": "24 months from mfg",
            "storage_instructions": "Store in airtight container away from moisture.",
            "country_of_origin": None,
            "claims": ["Cures diabetes naturally", "Boosts immunity 100%"],
        }
    },
]


def render_label_image(label_def: dict, out_path: Path) -> None:
    """Render a clean packaging label image."""
    img_w, img_h = 800, 600
    img = Image.new("RGB", (img_w, img_h), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([(20, 20), (img_w - 20, img_h - 20)], outline="#2C3E50", width=3)
    # Header background
    draw.rectangle([(23, 23), (img_w - 23, 80)], fill="#F8FAFC")

    # Header text
    draw.text((40, 35), label_def["title"], fill="#1E293B")
    draw.text((img_w - 250, 40), f"Cat: {label_def['category']}", fill="#64748B")

    # Veg / Non-Veg symbol if specified
    if label_def.get("is_veg") is True:
        # Green square with green circle
        sq_x, sq_y = img_w - 80, 100
        draw.rectangle([(sq_x, sq_y), (sq_x + 36, sq_y + 36)], outline="#16A34A", width=3)
        draw.ellipse([(sq_x + 8, sq_y + 8), (sq_x + 28, sq_y + 28)], fill="#16A34A")

    # Body lines
    y = 110
    for line in label_def["lines"][1:]:
        draw.text((40, y), line, fill="#0F172A")
        y += 36

    # Bottom footer with disclaimer
    draw.line([(25, img_h - 60), (img_w - 25, img_h - 60)], fill="#CBD5E1", width=1)
    draw.text((40, img_h - 45), "[FSSAI Compliance Verification Dataset — Sample Packaging Label]", fill="#94A3B8")

    img.save(out_path, format="PNG")
    print(f"Generated test image: {out_path.name}")


def main():
    ground_truth_data = {}
    for item in TEST_LABELS:
        out_file = IMAGES_DIR / item["filename"]
        render_label_image(item, out_file)
        ground_truth_data[item["filename"]] = {
            "title": item["title"],
            "category": item["category"],
            "ground_truth": item["ground_truth"],
        }

    gt_file = IMAGES_DIR / "ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth_data, f, indent=2)
    print(f"\nWrote ground truth for 10 test images to: {gt_file}")


if __name__ == "__main__":
    main()
