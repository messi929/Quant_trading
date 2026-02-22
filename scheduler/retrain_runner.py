"""재학습 스케줄러.

주간: Transformer 파인튜닝 (토요일, ~30분)
월간: Phase 2~6 전체 재학습 (매월 1일, ~2-3시간)

실행 방법:
  python scheduler/retrain_runner.py --mode weekly   # 주간 파인튜닝
  python scheduler/retrain_runner.py --mode monthly  # 월간 전체 재학습
  python scheduler/retrain_runner.py --mode check    # 재학습 필요 여부 확인
"""

import sys
import time
import shutil
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger
from utils.logger import setup_logger
from utils.device import set_seed
from tracking.trade_log import TradeLogger


class RetrainRunner:
    """모델 재학습 스케줄러."""

    def __init__(self, config_path: str = "config/live_config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        with open(self.cfg["model"]["config_path"], encoding="utf-8") as f:
            self.settings = yaml.safe_load(f)

        setup_logger(
            log_dir=self.settings["paths"]["logs"],
            level=self.settings["logging"]["level"],
        )
        set_seed(self.settings["training"]["seed"])

        self.retrain_cfg  = self.cfg["retrain"]
        self.model_dir    = Path(self.settings["paths"]["models"])
        self.backup_count = self.retrain_cfg.get("model_backup_count", 5)
        self.trade_logger = TradeLogger(self.cfg["logging"]["trade_log_db"])

    # ------------------------------------------------------------------
    # 재학습 필요 여부 판단
    # ------------------------------------------------------------------

    def should_retrain(self) -> dict:
        """재학습 필요 여부 자동 판단.

        Returns:
            {"weekly": bool, "monthly": bool, "reason": str}
        """
        accuracy = self.trade_logger.get_signal_accuracy(days=14)
        perf = self.trade_logger.get_performance_df(days=14)

        reasons = []
        force_weekly = False
        force_monthly = False

        # 1. 신호 정확도 저하
        dir_acc = accuracy.get("directional_accuracy")
        if dir_acc is not None and dir_acc < 0.48:
            reasons.append(f"신호 정확도 저하: {dir_acc:.1%} < 48%")
            force_weekly = True

        # 2. 최근 14일 성과 저하
        if not perf.empty and len(perf) >= 5:
            recent_sharpe = perf["sharpe_30d"].iloc[-1]
            if recent_sharpe < 0.5:
                reasons.append(f"Sharpe 저하: {recent_sharpe:.2f} < 0.5")
                force_weekly = True

        # 3. MDD 위험 수준
        if not perf.empty:
            mdd = perf["mdd_cumul"].min()
            if mdd < -0.10:
                reasons.append(f"MDD 경고: {mdd:.1%}")
                force_monthly = True

        reason_str = "; ".join(reasons) if reasons else "정기 재학습"
        logger.info(f"재학습 판단: weekly={force_weekly}, monthly={force_monthly}, 이유={reason_str}")

        return {
            "weekly":  force_weekly,
            "monthly": force_monthly,
            "reason":  reason_str,
        }

    # ------------------------------------------------------------------
    # 모델 백업
    # ------------------------------------------------------------------

    def _backup_models(self):
        """현재 모델 백업 (롤백 대비)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.model_dir / "backups" / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)

        for f in self.model_dir.glob("*.pt"):
            shutil.copy2(f, backup_dir / f.name)

        logger.info(f"모델 백업 완료: {backup_dir}")

        # 오래된 백업 삭제
        backups = sorted((self.model_dir / "backups").iterdir())
        while len(backups) > self.backup_count:
            oldest = backups.pop(0)
            shutil.rmtree(oldest)
            logger.info(f"오래된 백업 삭제: {oldest.name}")

    def rollback_model(self, backup_name: str = None):
        """백업에서 모델 복원."""
        backup_base = self.model_dir / "backups"
        if not backup_base.exists():
            logger.error("백업 디렉토리 없음")
            return

        if backup_name:
            backup_dir = backup_base / backup_name
        else:
            # 가장 최근 백업
            backups = sorted(backup_base.iterdir())
            if not backups:
                logger.error("백업 없음")
                return
            backup_dir = backups[-1]

        for f in backup_dir.glob("*.pt"):
            shutil.copy2(f, self.model_dir / f.name)

        logger.info(f"모델 롤백 완료: {backup_dir.name}")

    # ------------------------------------------------------------------
    # 주간 파인튜닝 (Transformer만)
    # ------------------------------------------------------------------

    def run_weekly(self, trigger: str = "weekly"):
        """Transformer 파인튜닝 (Phase 3만 재실행)."""
        logger.info("\n" + "=" * 60)
        logger.info(f"주간 파인튜닝 시작: {datetime.now():%Y-%m-%d %H:%M:%S}")
        logger.info("=" * 60)

        start_time = time.time()
        self._backup_models()

        try:
            from pipeline.train_pipeline import TrainPipeline

            pipeline = TrainPipeline(config_path=self.cfg["model"]["config_path"])
            results = pipeline.run(
                skip_collection=True,           # 기존 데이터 사용
                start_phase=self.retrain_cfg.get("weekly_start_phase", 3),
            )

            elapsed = int(time.time() - start_time)

            # 결과에서 Transformer 지표 추출
            tf_results = results.get("transformer", {})
            val_loss = tf_results.get("best_val_loss", 0)
            dir_acc  = tf_results.get("best_val_dir_acc", 0)

            self.trade_logger.log_retrain(
                phase_start=self.retrain_cfg.get("weekly_start_phase", 3),
                duration_sec=elapsed,
                val_loss=val_loss,
                dir_acc=dir_acc,
                trigger=trigger,
            )

            logger.info(
                f"주간 파인튜닝 완료: {elapsed//60}분 {elapsed%60}초 소요, "
                f"val_loss={val_loss:.4f}, dir_acc={dir_acc:.1%}"
            )

            # 성능 저하 시 자동 롤백
            if dir_acc < 0.48:
                logger.warning(f"파인튜닝 후 dir_acc={dir_acc:.1%} 저하 → 롤백")
                self.rollback_model()

        except Exception as e:
            logger.error(f"주간 파인튜닝 실패: {e}")
            logger.info("이전 모델로 롤백")
            self.rollback_model()
            raise

    # ------------------------------------------------------------------
    # 월간 전체 재학습
    # ------------------------------------------------------------------

    def run_monthly(self, trigger: str = "monthly"):
        """전체 모델 재학습 (Phase 2~6, 5년 롤링 윈도우)."""
        logger.info("\n" + "=" * 60)
        logger.info(f"월간 전체 재학습 시작: {datetime.now():%Y-%m-%d %H:%M:%S}")
        logger.info("=" * 60)

        start_time = time.time()
        self._backup_models()

        try:
            from pipeline.train_pipeline import TrainPipeline

            pipeline = TrainPipeline(config_path=self.cfg["model"]["config_path"])
            results = pipeline.run(
                skip_collection=False,          # 새 데이터 수집 포함
                start_phase=self.retrain_cfg.get("monthly_start_phase", 2),
            )

            elapsed = int(time.time() - start_time)
            tf_results = results.get("transformer", {})
            val_loss = tf_results.get("best_val_loss", 0)
            dir_acc  = tf_results.get("best_val_dir_acc", 0)

            self.trade_logger.log_retrain(
                phase_start=self.retrain_cfg.get("monthly_start_phase", 2),
                duration_sec=elapsed,
                val_loss=val_loss,
                dir_acc=dir_acc,
                trigger=trigger,
            )

            logger.info(
                f"월간 재학습 완료: {elapsed//3600}시간 {(elapsed%3600)//60}분 소요, "
                f"dir_acc={dir_acc:.1%}"
            )

        except Exception as e:
            logger.error(f"월간 재학습 실패: {e}")
            logger.info("이전 모델로 롤백")
            self.rollback_model()
            raise


def main():
    parser = argparse.ArgumentParser(description="재학습 스케줄러")
    parser.add_argument(
        "--mode",
        choices=["weekly", "monthly", "check", "rollback"],
        default="check",
        help="실행 모드",
    )
    parser.add_argument("--config", default="config/live_config.yaml")
    args = parser.parse_args()

    runner = RetrainRunner(config_path=args.config)

    if args.mode == "check":
        result = runner.should_retrain()
        print(f"재학습 필요: {result}")
    elif args.mode == "weekly":
        runner.run_weekly(trigger="manual")
    elif args.mode == "monthly":
        runner.run_monthly(trigger="manual")
    elif args.mode == "rollback":
        runner.rollback_model()


if __name__ == "__main__":
    main()
