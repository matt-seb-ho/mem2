from mem2.branches.router.llm import LlmRouter
from mem2.branches.router.nli import NliRouter
from mem2.branches.router.none import NoneRouter
from mem2.branches.router.threshold import ThresholdRouter

ROUTERS = {
    "none": NoneRouter,
    "threshold": ThresholdRouter,
    "llm": LlmRouter,
    "nli": NliRouter,
}
