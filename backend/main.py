
import os
import torch
import torch.nn as nn
import cv2
import numpy as np

from pathlib import Path
from io import BytesIO
from PIL import Image

from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from google import genai


# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

load_dotenv(PROJECT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(
    title="Diabetic Retinopathy Report Generation API",
    version="1.0"
)


# ==========================================
# 2. DEVICE CONFIGURATION
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 3. LOAD EFFICIENTNET-B0 MODEL
# ==========================================

model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)

MODEL_PATH = BASE_DIR / "models" / "efficientnet_b0_dr.pth"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()


# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

IMG_SIZE = 224

eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def crop_black(img, tol=7):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol

    if mask.sum() == 0:
        return img

    ys, xs = np.where(mask)

    return img[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]


# ==========================================
# 5. DR GRADES
# ==========================================

stages = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# ==========================================
# 6. GEMINI CLIENT
# ==========================================

def get_gemini_client():

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is missing from .env"
        )

    return genai.Client(api_key=GEMINI_API_KEY)


GEMINI_MODEL = "gemini-3.5-flash-lite"


# ==========================================
# 7. PROMPTING STRATEGIES
# ==========================================

def get_prompt(strategy, grade, stage, confidence):

    common_information = f"""
    CNN Prediction Information:

    DR Grade: {grade}
    DR Stage: {stage}
    CNN Confidence: {confidence}%

    Dataset: APTOS 2019 Diabetic Retinopathy.

    Generate an educational diabetic retinopathy
    report using only the prediction information
    provided above.

    Important:
    - Do not invent retinal image findings.
    - Do not claim that a doctor examined the patient.
    - Do not treat model confidence as clinical certainty.
    - Do not invent patient history or symptoms.
    - Do not prescribe medication or treatment.
    - Clearly state that this is an AI-generated
      research output, not a clinical diagnosis.
    """

    prompts = {

        "zero_shot": """
        Generate a clear and professional diabetic
        retinopathy report from the supplied CNN
        prediction.

        Include:
        1. Predicted DR grade
        2. Stage interpretation
        3. Model confidence
        4. Brief explanation
        5. General recommendation for professional
           eye-care evaluation.

        Do not use examples.
        """,

        "few_shot": """
        Generate a diabetic retinopathy report using
        the following example of the desired style.

        Example:
        Prediction: Grade 0
        Stage: No DR
        Report: The AI model classified the image
        as No DR. This is a model prediction and
        does not rule out eye disease.

        Now generate a report using the actual
        prediction information supplied.

        Do not copy the example prediction.
        """,

        "chain_of_thought": """
        Use a structured step-by-step assessment
        framework to organize the report.

        Consider:
        1. The supplied CNN grade.
        2. The corresponding DR stage.
        3. The reported model confidence.
        4. The limitations of AI-based classification.

        Provide only a concise explanation of the
        assessment, not private internal reasoning.

        End with a suitable professional
        evaluation recommendation.
        """,

        "template_constrained": """
        Generate the report using exactly this format:

        DIABETIC RETINOPATHY REPORT

        1. Predicted DR Grade:
        2. DR Stage:
        3. CNN Confidence:
        4. Prediction Interpretation:
        5. General Recommendation:
        6. Limitations:

        Keep all six sections.
        Do not add unsupported clinical findings.
        """,

        "safety_constrained": """
        Generate a safety-focused diabetic
        retinopathy report.

        Requirements:

        1. Clearly distinguish AI prediction from
           confirmed clinical diagnosis.
        2. Never claim that the image proves the
           presence or absence of disease.
        3. Do not invent retinal abnormalities.
        4. Do not recommend medication.
        5. Explain that model confidence is not
           equivalent to diagnostic certainty.
        6. Recommend professional eye-care
           assessment.
        7. Use cautious and medically responsible
           language.
        """
    }

    return common_information + prompts[strategy]


# ==========================================
# 8. GENERATE FIVE REPORTS
# ==========================================

def generate_all_reports(grade, stage, confidence):

    client = get_gemini_client()

    strategies = [
        "zero_shot",
        "few_shot",
        "chain_of_thought",
        "template_constrained",
        "safety_constrained"
    ]

    reports = {}

    for strategy in strategies:

        try:

            prompt = get_prompt(
                strategy,
                grade,
                stage,
                confidence
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            if not response.text:
                raise ValueError("Gemini returned an empty report")

            reports[strategy] = response.text

        except Exception as e:

            reports[strategy] = {
                "error": str(e)
            }

    return reports


# ==========================================
# 9. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message":
        "Diabetic Retinopathy Report Generation API is running",
        "status": "success"
    }


# ==========================================
# 10. CNN PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        if file.content_type not in [
            "image/jpeg",
            "image/png",
            "image/jpg"
        ]:
            raise HTTPException(
                status_code=400,
                detail="Please upload a JPEG or PNG image"
            )

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty"
            )

        img = Image.open(
            BytesIO(contents)
        ).convert("RGB")

        img = np.array(img)

        img = crop_black(img)

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        img_tensor = eval_transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():

            outputs = model(img_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        grade = prediction.item()

        return {
            "grade": grade,
            "stage": stages[grade],
            "confidence": round(
                confidence.item() * 100, 2
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# 11. GEMINI REPORT GENERATION ENDPOINT
# ==========================================

class ReportRequest(BaseModel):

    grade: int = Field(ge=0, le=4)

    stage: str

    confidence: float = Field(ge=0, le=100)


@app.post("/generate-reports")
def generate_reports(request: ReportRequest):

    expected_stage = stages[request.grade]

    if request.stage != expected_stage:
        raise HTTPException(
            status_code=400,
            detail="DR stage does not match the supplied grade"
        )

    reports = generate_all_reports(
        request.grade,
        request.stage,
        request.confidence
    )

    successful = [
        name for name, report in reports.items()
        if isinstance(report, str)
    ]

    return {
        "prediction": {
            "grade": request.grade,
            "stage": request.stage,
            "confidence": request.confidence
        },
        "reports": reports,
        "successful_reports": len(successful),
        "total_strategies": 5
    }
