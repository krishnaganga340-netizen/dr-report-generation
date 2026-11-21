
import os
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env file.")

client = genai.Client(api_key=api_key)

# Read CNN predictions
df = pd.read_csv("cnn_outputs_50.csv")

# Output file
output_file = "generated_reports_250.csv"

# Load previously generated reports
if os.path.exists(output_file):
    results_df = pd.read_csv(output_file)
    results = results_df.to_dict("records")
    completed = set(
        (str(row["case_id"]), str(row["prompt_style"]))
        for row in results
    )
    print("Previously saved reports:", len(results))
else:
    results = []
    completed = set()

# Five prompting strategies
prompt_styles = {
    "zero_shot": """
Write a short clinical-style diabetic retinopathy
screening report using the CNN information provided.
Include predicted stage, confidence and a brief
explanation. Do not invent patient details or findings.
""",

    "few_shot": """
Write a report following this example structure:

Predicted stage: [stage]
Model confidence: [confidence]
Explanation: The CNN model predicted [stage].
Disclaimer: AI-generated result; professional review
is required.

Use the current case values, not the example values.
Do not invent clinical findings.
""",

    "chain_of_thought": """
Carefully consider the supplied CNN prediction.
Write a brief report explaining the predicted grade,
stage and confidence. Provide only a concise explanation,
not private internal reasoning. Do not invent findings.
""",

    "template_constrained": """
Use exactly this report structure:

Diabetic Retinopathy Screening Report

1. Predicted DR Grade:
2. Predicted DR Stage:
3. Model Confidence:
4. Explanation:
5. Disclaimer:

Use only the supplied CNN information.
""",

    "safety_constrained": """
Generate a cautious screening report based only on
the supplied CNN prediction.

Do not claim a confirmed diagnosis.
Do not invent symptoms, retinal findings or patient
details. Clearly state that this is an AI-generated
prediction requiring review by an eye-care professional.
"""
}

# Generate missing reports only
for _, case in df.iterrows():

    for style, instruction in prompt_styles.items():

        case_id = str(case["case_id"])
        task = (case_id, style)

        # Skip reports already saved
        if task in completed:
            print(f"Skipping Case {case_id}: {style}")
            continue

        prompt = f"""
You are assisting with AI-based diabetic retinopathy
screening report generation.

CNN information:
- Case ID: {case["case_id"]}
- Predicted DR Grade: {case["cnn_grade"]}
- Predicted DR Stage: {case["cnn_stage"]}
- Model Confidence: {case["confidence"]}

Prompting strategy: {style}

Instructions:
{instruction}
"""

        print(f"Generating Case {case_id}: {style}")

        # Retry temporary server errors
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config={
                        "temperature": 0.0,
                        "max_output_tokens": 250
                    }
                )

                report = response.text or ""

                if not report.strip():
                    raise ValueError("Gemini returned an empty report.")

                results.append({
                    "case_id": case["case_id"],
                    "true_grade": case["true_grade"],
                    "cnn_grade": case["cnn_grade"],
                    "confidence": case["confidence"],
                    "prompt_style": style,
                    "report_text": report
                })

                completed.add(task)

                # Save after every successful report
                pd.DataFrame(results).to_csv(
                    output_file, index=False
                )

                print("Saved successfully. Total:", len(results))
                time.sleep(2)
                break

            except Exception as error:
                print("Temporary request failure:", error)

                if attempt == 4:
                    pd.DataFrame(results).to_csv(
                        output_file, index=False
                    )
                    print("Retry limit reached. Run the file again later.")
                    raise

                wait_time = 5 * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

print("\nExperiment completed!")
print("Total reports:", len(results))
print("Saved file:", output_file)
