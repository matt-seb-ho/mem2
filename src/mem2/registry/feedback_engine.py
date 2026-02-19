from mem2.branches.feedback_engine.gt_check import GroundTruthFeedbackEngine
from mem2.branches.feedback_engine.lcb_gt import LcbGroundTruthFeedbackEngine
from mem2.branches.feedback_engine.math_ps_gt import MathPsGroundTruthFeedbackEngine

FEEDBACK_ENGINES = {
    "gt_check": GroundTruthFeedbackEngine,
    "gt_check_v1": GroundTruthFeedbackEngine,
    "lcb_gt": LcbGroundTruthFeedbackEngine,
    "math_ps_gt": MathPsGroundTruthFeedbackEngine,
}
