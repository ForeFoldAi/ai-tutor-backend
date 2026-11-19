"""Phase 00 — architecture notes (not a runnable evaluator).

The evaluation framework is organized as numbered phases under
``evaluation/phases/phaseNN_*``. Each phase:

1. Declares ``phase_id``, ``feature``, optional ``depends_on``
2. Is registered with ``@register_phase``
3. Implements ``run(ctx) -> PhaseReport`` with PASS/FAIL ``CheckResult``s
4. Writes evidence under ``artifacts/<run_id>/<dataset>/<feature>/``

To add a new AI feature evaluator:

1. Create ``evaluation/phases/phase21_my_feature/__init__.py``
2. Implement a ``Phase`` subclass + ``@register_phase``
3. Import it from ``evaluation.core.registry.discover_phases``
4. Add golden JSON under ``goldens/<dataset>/my_feature.json``
5. Add pytest module under ``tests/my_feature/``
6. Enable in ``configs/phases.yaml`` / ``default.yaml``
"""
