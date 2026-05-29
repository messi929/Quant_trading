"""V4 Korea momentum engine (KOSDAQ multi-lb ensemble + 200d SMA gate + vol-target).

검증: full-cycle(2014-2026) survivorship-free Sharpe 0.66 / annual +10.5% / MDD -17%.
설계/검증 상세: docs/V4_DUAL_ENGINE_DESIGN.md §1.9 SPEC.

원칙 (CLAUDE.md 계승): 백테스트-라이브 단일 코드 경로. engine.py 의 순수 함수를
backtest 와 live 가 동일 호출.
"""
