import re
from datetime import datetime
from enum import StrEnum

from orm import Base
from sqlalchemy import DateTime, Enum, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


class McpStatus(StrEnum):
    disabled = "disabled"
    active = "active"
    unreachable = "unreachable"


_TRANSITIONS: dict[McpStatus, set[McpStatus]] = {
    McpStatus.disabled: {McpStatus.active},
    McpStatus.active: {McpStatus.disabled, McpStatus.unreachable},
    McpStatus.unreachable: {McpStatus.disabled, McpStatus.active},
}


def can_switch(src: McpStatus, dst: McpStatus) -> bool:
    return dst in _TRANSITIONS.get(src, set())


class McpIntegration(Base):
    __tablename__ = "mcp_integrations"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    url: Mapped[str]
    status: Mapped[McpStatus] = mapped_column(
        Enum(McpStatus, native_enum=False), default=McpStatus.disabled
    )
    allowed_tools: Mapped[list] = mapped_column(JSONB, default=list)
    tool_schemas: Mapped[dict] = mapped_column(JSONB, default=dict)
    auth: Mapped[dict | None] = mapped_column(JSONB)
    timeout_s: Mapped[int] = mapped_column(default=30)
    max_result_chars: Mapped[int] = mapped_column(default=4000)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"McpIntegration(id={self.id!r}, name={self.name!r}, status={self.status!r})"
