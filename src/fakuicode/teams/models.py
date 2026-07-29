"""Validated persistent models for Agent Teams."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
import re
from typing import Mapping
from uuid import UUID, uuid4


_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,30}[A-Za-z0-9])?\Z")
_SHA = re.compile(r"[0-9a-f]{40,64}\Z")


class BackendType(StrEnum):
    AUTO = "auto"
    IN_PROCESS = "in_process"
    SUBPROCESS = "subprocess"


class MemberStatus(StrEnum):
    STARTING = "starting"
    IDLE = "idle"
    PLANNING = "planning"
    WAITING_APPROVAL = "waiting_approval"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    DETACHED = "detached"


class TaskStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    PLANNING = "planning"
    WORKING = "working"
    INTEGRATING = "integrating"
    INTEGRATION_FAILED = "integration_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELETED = "deleted"


class TaskKind(StrEnum):
    READ_ONLY = "read_only"
    TASK_WORKTREE = "task_worktree"


class MessageType(StrEnum):
    TEXT = "text"
    PLAN_REQUEST = "plan_request"
    PLAN_REVIEW = "plan_review"
    TASK_EVENT = "task_event"
    IDLE_NOTICE = "idle_notice"
    PERMISSION_REQUEST = "permission_request"
    SHUTDOWN_REQUEST = "shutdown_request"
    SHUTDOWN_RESPONSE = "shutdown_response"


class TeamStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


def normalize_team_name(value: str) -> str:
    name = value.strip()
    if not _NAME.fullmatch(name):
        raise ValueError("名称必须为 1-32 位字母、数字或连字符，且不能以连字符开头或结尾。")
    return name.lower()


def _uuid(value: str, field: str) -> str:
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as error:
        raise ValueError(f"{field} 必须是 UUID。") from error


def _text(value: str, field: str, maximum: int) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{field} 不能为空。")
    if len(text) > maximum:
        raise ValueError(f"{field} 过长。")
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ActorContext:
    team_id: str
    member_id: str
    member_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "team_id", _uuid(self.team_id, "team_id"))
        object.__setattr__(self, "member_id", _uuid(self.member_id, "member_id"))
        object.__setattr__(self, "member_name", normalize_team_name(self.member_name))

    @property
    def workflow_capability(self) -> str:
        return f"team:{self.team_id}:workflow"


@dataclass(frozen=True)
class TeamRecord:
    team_id: str
    name: str
    lead_member_id: str
    lead_conversation_id: str
    repository_fingerprint: str
    target_branch: str
    target_sha: str
    status: TeamStatus
    created_at: str
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        name: str,
        lead_conversation_id: str,
        repository_fingerprint: str,
        target_branch: str,
        target_sha: str,
    ) -> TeamRecord:
        if _SHA.fullmatch(target_sha) is None:
            raise ValueError("target_sha 不是有效提交。")
        return cls(
            team_id=str(uuid4()),
            name=normalize_team_name(name),
            lead_member_id=str(uuid4()),
            lead_conversation_id=_text(lead_conversation_id, "lead_conversation_id", 200),
            repository_fingerprint=_text(repository_fingerprint, "repository_fingerprint", 512),
            target_branch=_text(target_branch, "target_branch", 255),
            target_sha=target_sha,
            status=TeamStatus.ACTIVE,
            created_at=_utc_now(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "team_id": self.team_id,
            "name": self.name,
            "lead_member_id": self.lead_member_id,
            "lead_conversation_id": self.lead_conversation_id,
            "repository_fingerprint": self.repository_fingerprint,
            "target_branch": self.target_branch,
            "target_sha": self.target_sha,
            "status": self.status.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TeamRecord:
        return cls(
            schema_version=int(value["schema_version"]),
            team_id=_uuid(str(value["team_id"]), "team_id"),
            name=normalize_team_name(str(value["name"])),
            lead_member_id=_uuid(str(value["lead_member_id"]), "lead_member_id"),
            lead_conversation_id=_text(
                str(value["lead_conversation_id"]), "lead_conversation_id", 200
            ),
            repository_fingerprint=_text(
                str(value["repository_fingerprint"]), "repository_fingerprint", 512
            ),
            target_branch=_text(str(value["target_branch"]), "target_branch", 255),
            target_sha=str(value["target_sha"]),
            status=TeamStatus(str(value["status"])),
            created_at=str(value["created_at"]),
        )


@dataclass(frozen=True)
class TeamMember:
    member_id: str
    name: str
    role: str
    agent_type: str
    profile: str
    backend: BackendType
    requires_plan_approval: bool
    conversation_id: str
    status: MemberStatus
    created_at: str
    current_task_id: str | None = None
    workspace: str | None = None
    runtime_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        name: str,
        role: str,
        agent_type: str = "general-purpose",
        profile: str,
        backend: BackendType,
        requires_plan_approval: bool,
        conversation_id: str,
        member_id: str | None = None,
    ) -> TeamMember:
        return cls(
            member_id=_uuid(member_id or str(uuid4()), "member_id"),
            name=normalize_team_name(name),
            role=_text(role, "role", 2_000),
            agent_type=_text(agent_type, "agent_type", 100),
            profile=_text(profile, "profile", 100),
            backend=backend,
            requires_plan_approval=requires_plan_approval,
            conversation_id=_text(conversation_id, "conversation_id", 200),
            status=MemberStatus.IDLE,
            created_at=_utc_now(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "member_id": self.member_id,
            "name": self.name,
            "role": self.role,
            "agent_type": self.agent_type,
            "profile": self.profile,
            "backend": self.backend.value,
            "requires_plan_approval": self.requires_plan_approval,
            "conversation_id": self.conversation_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "current_task_id": self.current_task_id,
            "workspace": self.workspace,
            "runtime_id": self.runtime_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TeamMember:
        current_task_id = value.get("current_task_id")
        if current_task_id is not None:
            current_task_id = _uuid(str(current_task_id), "current_task_id")
        return cls(
            member_id=_uuid(str(value["member_id"]), "member_id"),
            name=normalize_team_name(str(value["name"])),
            role=_text(str(value["role"]), "role", 2_000),
            agent_type=_text(
                str(value.get("agent_type", "general-purpose")), "agent_type", 100
            ),
            profile=_text(str(value["profile"]), "profile", 100),
            backend=BackendType(str(value["backend"])),
            requires_plan_approval=bool(value["requires_plan_approval"]),
            conversation_id=_text(str(value["conversation_id"]), "conversation_id", 200),
            status=MemberStatus(str(value["status"])),
            created_at=str(value["created_at"]),
            current_task_id=current_task_id,
            workspace=None if value.get("workspace") is None else str(value["workspace"]),
            runtime_id=None if value.get("runtime_id") is None else str(value["runtime_id"]),
        )


@dataclass(frozen=True)
class TeamTask:
    task_id: str
    title: str
    description: str
    created_by: str
    blocked_by: tuple[str, ...]
    kind: TaskKind
    status: TaskStatus
    assignee_id: str | None
    revision: int
    created_at: str
    updated_at: str
    result_summary: str = ""
    base_sha: str | None = None
    completion_sha: str | None = None
    plan_request_id: str | None = None
    plan_revision: int = 0
    plan_approved: bool = False
    worktree_branch: str | None = None
    workspace: str | None = None
    verification_summary: str = ""
    integration_sha: str | None = None

    @classmethod
    def create(
        cls,
        *,
        title: str,
        description: str,
        created_by: str,
        blocked_by: tuple[str, ...] = (),
        kind: TaskKind = TaskKind.TASK_WORKTREE,
    ) -> TeamTask:
        now = _utc_now()
        return cls(
            task_id=str(uuid4()),
            title=_text(title, "title", 500),
            description=description.strip(),
            created_by=_text(created_by, "created_by", 100),
            blocked_by=tuple(_uuid(item, "blocked_by") for item in blocked_by),
            kind=kind,
            status=TaskStatus.PENDING,
            assignee_id=None,
            revision=1,
            created_at=now,
            updated_at=now,
        )

    def revise(self, **changes: object) -> TeamTask:
        return replace(
            self,
            **changes,
            revision=self.revision + 1,
            updated_at=_utc_now(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "created_by": self.created_by,
            "blocked_by": list(self.blocked_by),
            "kind": self.kind.value,
            "status": self.status.value,
            "assignee_id": self.assignee_id,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result_summary": self.result_summary,
            "base_sha": self.base_sha,
            "completion_sha": self.completion_sha,
            "plan_request_id": self.plan_request_id,
            "plan_revision": self.plan_revision,
            "plan_approved": self.plan_approved,
            "worktree_branch": self.worktree_branch,
            "workspace": self.workspace,
            "verification_summary": self.verification_summary,
            "integration_sha": self.integration_sha,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TeamTask:
        blocked = value.get("blocked_by", ())
        if not isinstance(blocked, list):
            raise ValueError("blocked_by 必须是数组。")
        assignee = value.get("assignee_id")
        return cls(
            task_id=_uuid(str(value["task_id"]), "task_id"),
            title=_text(str(value["title"]), "title", 500),
            description=str(value.get("description", "")),
            created_by=_text(str(value["created_by"]), "created_by", 100),
            blocked_by=tuple(_uuid(str(item), "blocked_by") for item in blocked),
            kind=TaskKind(str(value["kind"])),
            status=TaskStatus(str(value["status"])),
            assignee_id=None if assignee is None else _uuid(str(assignee), "assignee_id"),
            revision=int(value["revision"]),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            result_summary=str(value.get("result_summary", "")),
            base_sha=None if value.get("base_sha") is None else str(value["base_sha"]),
            completion_sha=(
                None if value.get("completion_sha") is None else str(value["completion_sha"])
            ),
            plan_request_id=(
                None
                if value.get("plan_request_id") is None
                else _uuid(str(value["plan_request_id"]), "plan_request_id")
            ),
            plan_revision=int(value.get("plan_revision", 0)),
            plan_approved=bool(value.get("plan_approved", False)),
            worktree_branch=(
                None
                if value.get("worktree_branch") is None
                else str(value["worktree_branch"])
            ),
            workspace=None if value.get("workspace") is None else str(value["workspace"]),
            verification_summary=str(value.get("verification_summary", "")),
            integration_sha=(
                None
                if value.get("integration_sha") is None
                else str(value["integration_sha"])
            ),
        )


@dataclass(frozen=True)
class TeamMessage:
    message_id: str
    message_type: MessageType
    sender_id: str
    sender_name: str
    recipient_id: str
    recipient_name: str
    body: str
    summary: str
    created_at: str
    read: bool = False
    correlation_id: str | None = None
    payload: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "recipient_id": self.recipient_id,
            "recipient_name": self.recipient_name,
            "body": self.body,
            "summary": self.summary,
            "created_at": self.created_at,
            "read": self.read,
            "correlation_id": self.correlation_id,
            "payload": dict(self.payload) if self.payload is not None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object], *, read: bool | None = None) -> TeamMessage:
        payload = value.get("payload")
        if payload is not None and not isinstance(payload, Mapping):
            raise ValueError("消息 payload 必须是对象。")
        return cls(
            message_id=_uuid(str(value["message_id"]), "message_id"),
            message_type=MessageType(str(value["message_type"])),
            sender_id=_uuid(str(value["sender_id"]), "sender_id"),
            sender_name=normalize_team_name(str(value["sender_name"])),
            recipient_id=_uuid(str(value["recipient_id"]), "recipient_id"),
            recipient_name=normalize_team_name(str(value["recipient_name"])),
            body=str(value["body"]),
            summary=str(value["summary"]),
            created_at=str(value["created_at"]),
            read=bool(value.get("read", False)) if read is None else read,
            correlation_id=(
                None if value.get("correlation_id") is None else str(value["correlation_id"])
            ),
            payload=payload,
        )
