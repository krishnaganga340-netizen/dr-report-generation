
import os
import json
import requests
from pathlib import Path
from datetime import datetime

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "dr_test_images"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# FastAPI server
BASE_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{BASE_URL}/predict"
REPORT_URL = f"{BASE_URL}/generate-reports"

# Find all PNG images
image_files = sorted(IMAGE_DIR.glob("*.png"))[:1]

print(f"Found {len(image_files)} test images.")

all_results = []

for index, image_path in enumerate(image_files, start=1):
    print(f"\nProcessing {index}/{len(image_files)}: {image_path.name}")

    try:
        # Step 1: CNN prediction
        with open(image_path, "rb") as image_file:
            response = requests.post(
                PREDICT_URL,
                files={"file": (image_path.name, image_file, "image/png")},
                timeout=120
            )

        response.raise_for_status()
        prediction = response.json()

        # Step 2: Gemini report generation
        report_payload = {
            "grade": prediction["grade"],
            "stage": prediction["stage"],
            "confidence": prediction["confidence"]
        }

        report_response = requests.post(
            REPORT_URL,
            json=report_payload,
            timeout=300
        )

        report_response.raise_for_status()
        report_data = report_response.json()

        # Save one image's complete result
        result = {
            "test_case": index,
            "image_name": image_path.name,
            "prediction": prediction,
            "reports": report_data.get("reports", {}),
            "successful_reports": report_data.get("successful_reports", 0),
            "total_strategies": report_data.get("total_strategies", 5)
        }

        all_results.append(result)

        output_file = RESULTS_DIR / f"test_case_{index:02d}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print("CNN prediction:", prediction)
        print("Successful reports:", result["successful_reports"])

    except Exception as error:
        print(f"Error processing {image_path.name}: {error}")

# Save combined results
summary_file = RESULTS_DIR / "all_experiment_results.json"

with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print("\nExperiment completed.")
print("Images processed:", len(all_results))
print(
    "Total successful reports:",
    sum(r["successful_reports"] for r in all_results)
)
print("Results saved in:", RESULTS_DIR)
