import logging
import pdfplumber
import json
from typing import List, Dict
from groq import Groq
from data.schema import ClaimItem, LLMBillResponse
from config import settings
import pytesseract
import base64
import io

# Bind Tesseract binary path from config (overridable via .env)
pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

logger = logging.getLogger(__name__)

# Maximum PDF file size (20MB)
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024


class BillParser:
    def __init__(self):
        self.groq_client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None

    def parse(self, file_path: str) -> Dict:
        logger.info(f"Extracting raw text from bill: {file_path}")
        raw_text = ""

        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted and len(extracted.strip()) > 10:
                        raw_text += extracted + "\n"
                    else:
                        logger.warning("No digital text found on page. Cascading to Tesseract OCR...")
                        try:
                            pil_img = page.to_image(resolution=300).original
                            ocr_text = pytesseract.image_to_string(pil_img)
                            raw_text += ocr_text + "\n"
                        except Exception as ocr_e:
                            logger.warning(f"Tesseract missing/failed. Cascading to Groq Vision LLM! Error: {ocr_e}")
                            if self.groq_client:
                                buffered = io.BytesIO()
                                pil_img.save(buffered, format="PNG")
                                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

                                vis_res = self.groq_client.chat.completions.create(
                                    messages=[{
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": "Extract all text from this hospital bill image exactly as it appears."},
                                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                        ]
                                    }],
                                    model="llama-3.2-11b-vision-preview"
                                )
                                raw_text += vis_res.choices[0].message.content + "\n"

        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise ValueError(f"PDF Reading/OCR failed: {str(e)}")

        if not self.groq_client:
            raise ValueError("Groq API key is missing. Extraction aborted.")

        prompt = f"""You are an expert medical bill extractor. Given the OCR text of this hospital bill, extract patient details and all line items charged.
        
        OCR Text:
        {raw_text[:8000]}
        
        Return STRICTLY a JSON object with three keys:
        "patient_id": (str, name or ID),
        "admission_date": (str, date or N/A),
        "items": array of objects with keys: "item_id" (str), "code_type" (str), "code" (str), "description" (str), "qty" (int), "unit_cost" (float), "total" (float, MUST BE A NEGATIVE NUMBER if it is a discount or deduction!).
        """

        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)

            # ── Pydantic Validation ──
            validated = LLMBillResponse(**parsed)

            # ── Post-Processing & Cross-Verification ──
            output_items = []
            for idx, item in enumerate(validated.items):
                # Regenerate sequential item_ids to prevent LLM duplicates
                item_id = f"B{idx + 1}"

                qty = item.qty if item.qty and item.qty > 0 else 1
                unit_cost = item.unit_cost or 0.0
                total = item.total or 0.0

                # Cross-check: total ≈ qty × unit_cost (within 5% tolerance)
                expected_total = qty * unit_cost
                if unit_cost > 0 and total != 0:
                    if expected_total != 0 and abs(total - expected_total) / abs(expected_total) > 0.05:
                        logger.warning(
                            f"Bill item '{item.description}': total={total} != qty*unit_cost={expected_total}. "
                            f"Using LLM's total (it may include tax/rounding)."
                        )

                output_items.append(ClaimItem(
                    item_id=item_id,
                    code_type=item.code_type or "NONE",
                    code=item.code or "N/A",
                    description=item.description or "Unknown",
                    qty=qty,
                    unit_cost=unit_cost,
                    total=total
                ))

            logger.info(f"Bill parsed: {len(output_items)} items extracted and validated.")

            return {
                "patient_id": validated.patient_id or "UNKNOWN",
                "admission_date": validated.admission_date or "UNKNOWN",
                "items": output_items
            }
        except Exception as e:
            logger.error(f"LLM parsing error: {str(e)}")
            raise ValueError(f"Groq LLM parsing failed: {str(e)}")
