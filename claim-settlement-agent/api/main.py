from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import sqlite3
import json
import uuid
import os
import time
import tempfile
import pathlib

from ingestion.bill_parser import BillParser
from parser.policy_indexer import PolicyIndexer
from engine.reconciler import Reconciler
from engine.rule_engine import RuleEngine
from engine.micro_compiler import MicroCompiler
from engine.exclusion_matcher import ExclusionMatcher
from engine.decision import DecisionMaker
from data.schema import DecisionResult

app = FastAPI(title="Insurance Claim Settlement Agent")

# ─── Database ─────────────────────────────────────────────────────────
DB_PATH = "claims.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS claims (id TEXT PRIMARY KEY, data TEXT)''')

init_db()

# ─── Maximum upload size (20MB) ───────────────────────────────────────
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# ─── Frontend ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Adjudicator AI | Claim Settlement</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --primary-hover: #4f46e5;
                --bg-color: #0f172a;
                --glass-bg: rgba(30, 41, 59, 0.7);
                --glass-border: rgba(255, 255, 255, 0.1);
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
            }
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #020617 0%, #0f172a 100%);
                color: var(--text-main);
                min-height: 100vh;
                margin: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                overflow-x: hidden;
            }
            
            .glow-circle {
                position: absolute;
                width: 600px;
                height: 600px;
                background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, rgba(0,0,0,0) 70%);
                top: -200px;
                left: -200px;
                z-index: 0;
                pointer-events: none;
            }
            
            header {
                width: 100%;
                padding: 2rem 0;
                text-align: center;
                z-index: 1;
            }
            h1 {
                font-family: 'Outfit', sans-serif;
                font-size: 3rem;
                font-weight: 700;
                background: linear-gradient(to right, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0 0 0.5rem 0;
                letter-spacing: -1px;
            }
            p.subtitle {
                color: var(--text-muted);
                font-size: 1.1rem;
                margin-top: 0;
            }
            
            main {
                width: 100%;
                max-width: 800px;
                z-index: 1;
                display: flex;
                flex-direction: column;
                gap: 2rem;
                padding: 0 1rem 4rem 1rem;
            }

            .glass-panel {
                background: var(--glass-bg);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid var(--glass-border);
                border-radius: 24px;
                padding: 2.5rem;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .glass-panel:hover {
                transform: translateY(-2px);
                box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.6);
            }

            .upload-group {
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
                margin-bottom: 2rem;
            }
            
            .file-input-wrapper {
                position: relative;
                width: 100%;
            }
            
            .file-label {
                display: block;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: #e2e8f0;
            }

            input[type="file"] {
                width: 100%;
                padding: 1rem;
                background: rgba(15, 23, 42, 0.6);
                border: 1px dashed var(--primary);
                border-radius: 12px;
                color: var(--text-main);
                cursor: pointer;
                transition: all 0.2s ease;
            }
            input[type="file"]:hover {
                background: rgba(30, 41, 59, 0.8);
                border-color: #818cf8;
            }

            button {
                width: 100%;
                padding: 1.2rem;
                border: none;
                border-radius: 12px;
                background: linear-gradient(135deg, var(--primary), var(--primary-hover));
                color: white;
                font-family: 'Inter', sans-serif;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.4);
            }
            button:active {
                transform: translateY(0);
            }

            .loader {
                display: none;
                text-align: center;
                margin-top: 1rem;
                color: var(--primary);
                font-weight: 600;
                animation: pulse 1.5s infinite ease-in-out;
            }
            @keyframes pulse {
                0% { opacity: 0.6; transform: scale(0.98); }
                50% { opacity: 1; transform: scale(1.02); }
                100% { opacity: 0.6; transform: scale(0.98); }
            }

            #result-panel {
                display: none;
                margin-top: 2rem;
            }
            .status-badge {
                display: inline-block;
                padding: 0.5rem 1rem;
                border-radius: 9999px;
                font-weight: 700;
                font-size: 0.9rem;
                text-transform: uppercase;
                margin-bottom: 1rem;
            }
            .status-APPROVED { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
            .status-PARTIAL { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid #eab308; }
            .status-REJECTED { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
            
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 1rem;
                margin-bottom: 2rem;
            }
            .metric-box {
                background: rgba(15, 23, 42, 0.6);
                padding: 1.5rem;
                border-radius: 16px;
                border: 1px solid var(--glass-border);
            }
            .metric-label { font-size: 0.9rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px;}
            .metric-value { font-size: 2rem; font-family: 'Outfit', sans-serif; font-weight: 700; color: white; margin-top: 0.5rem;}
            
            .citations-box {
                background: rgba(239, 68, 68, 0.05);
                border-left: 4px solid #ef4444;
                padding: 1.5rem;
                border-radius: 0 12px 12px 0;
            }
            .citations-box h3 { margin: 0 0 1rem 0; color: #f87171; font-family: 'Outfit', sans-serif;}
            .citation-item { margin-bottom: 0.75rem; line-height: 1.6; color: #e2e8f0; font-size: 0.95rem; }

            .pipeline-info {
                margin-top: 1.5rem;
                padding: 1rem;
                background: rgba(99, 102, 241, 0.08);
                border: 1px solid rgba(99, 102, 241, 0.2);
                border-radius: 12px;
                font-size: 0.85rem;
                color: var(--text-muted);
            }
            .pipeline-info strong { color: #818cf8; }
        </style>
    </head>
    <body>
        <div class="glow-circle"></div>
        
        <header>
            <h1>Transparent AI Adjudicator</h1>
            <p class="subtitle">Deterministic + LLM Hybrid Pipeline — Auditable & Legally Citable</p>
        </header>

        <main>
            <div class="glass-panel">
                <form id="uploadForm">
                    <div class="upload-group">
                        <div class="file-input-wrapper">
                            <label class="file-label">📄 Upload Scanned Hospital Bill (PDF/Img)</label>
                            <input type="file" id="billPdf" required>
                        </div>
                        <div class="file-input-wrapper">
                            <label class="file-label">⚖️ Upload Insurance Policy (PDF)</label>
                            <input type="file" id="policyPdf" required>
                        </div>
                    </div>
                    <button type="submit" id="submitBtn">Analyze & Adjudicate Claim</button>
                </form>
                <div class="loader" id="loader">⚙️ Running hybrid deterministic + LLM pipeline...</div>
            </div>

            <div class="glass-panel" id="result-panel">
                <div id="badge" class="status-badge"></div>
                
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">Total Claimed</div>
                        <div class="metric-value" id="claimedVal">₹0.00</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Total Approved</div>
                        <div class="metric-value" id="approvedVal">₹0.00</div>
                    </div>
                </div>

                <div class="citations-box" id="citationsBox" style="display: none;">
                    <h3>Official Citations</h3>
                    <div id="citationList"></div>
                </div>

                <div class="pipeline-info" id="pipelineInfo" style="display: none;"></div>
            </div>
        </main>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const billFile = document.getElementById('billPdf').files[0];
                const policyFile = document.getElementById('policyPdf').files[0];
                const btn = document.getElementById('submitBtn');
                const loader = document.getElementById('loader');
                const resultPanel = document.getElementById('result-panel');
                
                const formData = new FormData();
                formData.append('bill_pdf', billFile);
                formData.append('policy_pdf', policyFile);
                
                btn.style.display = 'none';
                loader.style.display = 'block';
                resultPanel.style.display = 'none';
                
                try {
                    const res = await fetch('/claim', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await res.json();
                    
                    if (!res.ok) throw new Error(data.detail || "Server error");
                    
                    const badge = document.getElementById('badge');
                    badge.className = `status-badge status-${data.overall_status}`;
                    badge.textContent = data.overall_status;
                    
                    document.getElementById('claimedVal').textContent = `₹${data.total_claimed.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
                    document.getElementById('approvedVal').textContent = `₹${data.total_approved.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
                    
                    const citationsBox = document.getElementById('citationsBox');
                    const citationList = document.getElementById('citationList');
                    citationList.innerHTML = '';
                    
                    if (data.citations && data.citations.length > 0) {
                        citationsBox.style.display = 'block';
                        data.citations.forEach(c => {
                            const p = document.createElement('div');
                            p.className = 'citation-item';
                            p.innerHTML = `<strong>•</strong> ${c}`;
                            citationList.appendChild(p);
                        });
                    } else {
                        citationsBox.style.display = 'none';
                    }

                    // Pipeline metadata
                    const pipelineInfo = document.getElementById('pipelineInfo');
                    if (data.pipeline_metadata) {
                        const m = data.pipeline_metadata;
                        let statsHtml = `
                            <strong>Pipeline Stats:</strong> 
                            Deterministic: ${m.deterministic_exclusions || 0} | 
                            LLM exclusions: ${m.llm_exclusions || 0} | 
                            LLM approvals: ${m.llm_approvals || 0} | 
                            LLM errors: ${m.llm_errors || 0} | 
                            Model: ${m.llm_model_used || 'N/A'} | 
                            Time: ${m.processing_time_ms || 'N/A'}ms
                        `;
                        if (m.error_details && m.error_details.length > 0) {
                            statsHtml += '<br><strong style="color:#f87171">Errors:</strong> ' + m.error_details.join(' | ');
                        }
                        pipelineInfo.innerHTML = statsHtml;
                        pipelineInfo.style.display = 'block';
                    }
                    
                    resultPanel.style.display = 'block';
                    
                } catch (err) {
                    alert('Analysis Failed: ' + err.message);
                } finally {
                    btn.style.display = 'block';
                    loader.style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# ─── Claim Processing Pipeline ────────────────────────────────────────

@app.post("/claim", response_model=DecisionResult)
async def process_claim(bill_pdf: UploadFile = File(...), policy_pdf: UploadFile = File(...)):
    # File size validation
    bill_content = await bill_pdf.read()
    policy_content = await policy_pdf.read()

    if len(bill_content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Bill PDF exceeds 20MB limit.")
    if len(policy_content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Policy PDF exceeds 20MB limit.")

    # Save files temporarily using tempfile for cross-platform safety
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_bill:
        tmp_bill.write(bill_content)
        bill_path = tmp_bill.name
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_policy:
        tmp_policy.write(policy_content)
        policy_path = tmp_policy.name

    try:
        start_time = time.time()

        # ── Step 1: Parse Bill ──
        parser = BillParser()
        bill_data = parser.parse(bill_path)
        items = bill_data.get("items", [])

        if not items:
            raise HTTPException(
                status_code=400,
                detail="Extraction returned 0 items. Raw text may be empty or LLM failed."
            )

        # ── Step 2: Index Policy (BM25 + Semantic Hybrid) ──
        indexer = PolicyIndexer()
        indexer.parse_and_index(policy_path)

        # ── Step 3: Extract Deterministic Exclusions ──
        exclusion_matcher = ExclusionMatcher()
        exclusion_matcher.extract_exclusions_from_clauses(indexer.clauses)

        # ── Step 4: Reconcile Items ──
        reconciler = Reconciler(indexer.clauses)
        reconciled_items = [reconciler.reconcile_item(item) for item in items]

        # ── Step 5: 2-Tier Adjudication (Deterministic + LLM RAG) ──
        compiler = MicroCompiler(indexer, exclusion_matcher)
        dsl_rules, item_mapping = compiler.compile_rules_for_bill(reconciled_items)

        # ── Step 6: Rule Engine (Pure Math) ──
        rule_engine = RuleEngine()
        evaluated_items = rule_engine.evaluate(
            items=reconciled_items,
            dsl_rules=dsl_rules,
            mapping=item_mapping,
            claim_metadata={"ytd_approved": 0.0}
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # ── Step 7: Final Decision ──
        pipeline_metadata = {
            **compiler.pipeline_stats,
            "processing_time_ms": elapsed_ms,
            "total_items": len(items),
            "exclusions_extracted": len(exclusion_matcher.exclusion_items),
            "policy_clauses_indexed": len(indexer.clauses),
        }

        decision_maker = DecisionMaker(indexer.clauses)
        claim_id = str(uuid.uuid4())
        final_decision = decision_maker.generate_result(
            claim_id=claim_id,
            patient_id=bill_data.get("patient_id", "Unknown"),
            admission_date=bill_data.get("admission_date", "Unknown"),
            items=evaluated_items,
            pipeline_metadata=pipeline_metadata
        )

        # ── Step 8: Log to DB ──
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO claims VALUES (?, ?)",
                (claim_id, json.dumps(
                    final_decision.__dict__,
                    default=lambda x: str(x) if not isinstance(x, (dict, list, str, int, float, type(None))) else x
                ))
            )

        return final_decision

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(bill_path):
            os.remove(bill_path)
        if os.path.exists(policy_path):
            os.remove(policy_path)


@app.get("/claim/{claim_id}")
async def get_claim(claim_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT data FROM claims WHERE id = ?", (claim_id,))
        row = c.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Claim not found")

    return json.loads(row[0])
