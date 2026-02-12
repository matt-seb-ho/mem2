from __future__ import annotations

from .entities import EventRecord, utc_now_iso


class LifecycleLogger:
    def __init__(self) -> None:
        self.events: list[EventRecord] = []

    def emit(self, stage: str, component: str, level: str, message: str, payload: dict | None = None) -> None:
        self.events.append(
            EventRecord(
                ts_utc=utc_now_iso(),
                stage=stage,
                component=component,
                level=level,
                message=message,
                payload=payload or {},
            )
        )
