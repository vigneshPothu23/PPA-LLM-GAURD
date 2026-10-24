"""PPA Guard v5 - independent application namespace.

Engine 1 (polymorphic prompt assembly), Engine 2 (transformer prompt-injection detector with its own training,
evaluation and inference code), the policy engine, RBAC, the security workflow, the Python SDK and the Streamlit UI.
Nothing in this package imports the earlier `ppa_guard` package; that implementation stays in the repository only for
comparison (enforced by tests/v5/test_independence.py).
"""

__version__ = "5.0.0"
