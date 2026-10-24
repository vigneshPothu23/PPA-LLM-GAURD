"""PPA Guard v5 - prompt-injection defence console (Streamlit Community Cloud entry point).

    sign-in -> role + organization + application scope -> Engine 2 (transformer detector, ONNX, CPU)
    -> policy (ALLOW / WARN / BLOCK) -> Engine 1 (polymorphic prompt assembly) -> audit record -> result

Local demo:   PPAGUARD_DEMO_MODE=1 streamlit run streamlit_app.py
Production:   see docs/v5/DEPLOY.md (OIDC sign-in, PostgreSQL, model selection).
The previous v4 console is preserved in legacy/console_v4_app.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ppaguard.ui.app import main  # noqa: E402

main(ROOT)
