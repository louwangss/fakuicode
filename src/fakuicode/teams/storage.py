"""SQLite persistence for authoritative Team coordination state."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import stat
from uuid import UUID, uuid4

from fakuicode.teams.locking import KernelFileLock
from fakuicode.teams.models import (
    ActorContext,
    MessageType,
    TaskStatus,
    TeamMember,
    TeamMessage,
    TeamRecord,
    TeamTask,
    normalize_team_name,
)


_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
_MAX_MESSAGE_BODY_CHARS = 20_000
_MAX_SUMMARY_CHARS = 2_000
_SCHEMA_VERSION = 1


class TeamStoreError(RuntimeError):
    pass


class DuplicateTeamError(TeamStoreError):
    pass


class TeamNotFoundError(TeamStoreError):
    pass


class TeamStore:
    """Persist Team state atomically while retaining legacy JSON as a backup."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.database_path = self.root / "teams.sqlite3"
        self._ensure_root()
        with KernelFileLock(self.root / ".migration.lock"):
            self._initialize_schema()
            self._import_legacy_teams()

    def create_team(
        self,
        *,
        name: str,
        lead_conversation_id: str,
        repository_fingerprint: str,
        target_branch: str,
        target_sha: str,
    ) -> TeamRecord:
        team = TeamRecord.create(
            name=normalize_team_name(name),
            lead_conversation_id=lead_conversation_id,
            repository_fingerprint=repository_fingerprint,
            target_branch=target_branch,
            target_sha=target_sha,
        )
        try:
            with self._transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO teams(team_id, name, data_json)
                    VALUES (?, ?, ?)
                    """,
                    (team.team_id, team.name, _encode(team.to_dict())),
                )
        except sqlite3.IntegrityError as error:
            raise DuplicateTeamError(f"团队 '{team.name}' 已存在。") from error
        return team

    def list_teams(self) -> tuple[TeamRecord, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT data_json FROM teams ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return tuple(TeamRecord.from_dict(_decode(row["data_json"])) for row in rows)

    def get_team(self, team_id_or_name: str) -> TeamRecord:
        with self._connection() as connection:
            return self._get_team(connection, team_id_or_name)

    def add_member(self, team_id: str, member: TeamMember) -> None:
        try:
            with self._transaction() as connection:
                team = self._get_team(connection, team_id)
                self._insert_member(connection, team.team_id, member)
        except sqlite3.IntegrityError as error:
            raise ValueError(f"成员名 '{member.name}' 或成员 ID 已注册。") from error

    def save_member(self, team_id: str, member: TeamMember) -> None:
        try:
            with self._transaction() as connection:
                team = self._get_team(connection, team_id)
                self._save_member(connection, team.team_id, member)
        except sqlite3.IntegrityError as error:
            raise ValueError(f"成员名 '{member.name}' 已注册。") from error

    def update_member(
        self,
        team_id: str,
        member_id: str,
        updater: Callable[[TeamMember], TeamMember],
    ) -> TeamMember:
        """Atomically update one member from its latest persisted state."""

        normalized = str(UUID(member_id))
        try:
            with self._transaction() as connection:
                team = self._get_team(connection, team_id)
                current = self._get_member(connection, team.team_id, normalized)
                updated = updater(current)
                if updated.member_id != current.member_id:
                    raise ValueError("成员更新不能改变成员 ID。")
                if updated != current:
                    self._save_member(connection, team.team_id, updated)
                return updated
        except sqlite3.IntegrityError as error:
            raise ValueError(f"成员名 '{normalized}' 已注册。") from error

    def list_members(self, team_id: str) -> tuple[TeamMember, ...]:
        with self._connection() as connection:
            team = self._get_team(connection, team_id)
            return self._list_members(connection, team.team_id)

    def resolve_member(self, team_id: str, name_or_id: str) -> str:
        with self._connection() as connection:
            team = self._get_team(connection, team_id)
            return self._resolve_member(connection, team.team_id, name_or_id)

    def send_message(
        self,
        actor: ActorContext,
        *,
        to: str,
        body: str,
        summary: str,
        message_type: MessageType = MessageType.TEXT,
        payload: Mapping[str, object] | None = None,
        correlation_id: str | None = None,
    ) -> TeamMessage:
        with self._transaction() as connection:
            team = self._get_team(connection, actor.team_id)
            message = self._build_message(
                connection,
                team,
                actor,
                to=to,
                body=body,
                summary=summary,
                message_type=message_type,
                payload=payload,
                correlation_id=correlation_id,
            )
            self._insert_message(connection, team.team_id, message)
            return message

    def commit_workflow_transition(
        self,
        actor: ActorContext,
        *,
        task: TeamTask,
        member: TeamMember,
        to: str,
        body: str,
        summary: str,
        message_type: MessageType,
        payload: Mapping[str, object] | None = None,
        correlation_id: str | None = None,
    ) -> TeamMessage:
        """Commit a task, member, and workflow message as one durable transition."""

        try:
            with self._transaction() as connection:
                team = self._get_team(connection, actor.team_id)
                self._validate_actor(connection, team.team_id, actor)
                self._save_task(connection, team.team_id, task)
                self._save_member(connection, team.team_id, member)
                message = self._build_message(
                    connection,
                    team,
                    actor,
                    to=to,
                    body=body,
                    summary=summary,
                    message_type=message_type,
                    payload=payload,
                    correlation_id=correlation_id,
                )
                self._insert_message(connection, team.team_id, message)
                return message
        except sqlite3.IntegrityError as error:
            raise ValueError("工作流状态违反 Team 数据约束。") from error

    def list_messages(
        self,
        team_id: str,
        member_id: str,
        *,
        unread_only: bool = False,
        limit: int = 200,
    ) -> tuple[TeamMessage, ...]:
        if limit < 1 or limit > 200:
            raise ValueError("邮箱分页 limit 必须在 1 到 200 之间。")
        normalized = str(UUID(member_id))
        with self._connection() as connection:
            team = self._get_team(connection, team_id)
            self._get_member(connection, team.team_id, normalized)
            unread_clause = "AND is_read = 0" if unread_only else ""
            rows = connection.execute(
                f"""
                SELECT data_json, is_read
                FROM messages
                WHERE team_id = ? AND recipient_id = ? {unread_clause}
                ORDER BY sequence DESC
                LIMIT ?
                """,
                (team.team_id, normalized, limit),
            ).fetchall()
        rows.reverse()
        return tuple(
            TeamMessage.from_dict(_decode(row["data_json"]), read=bool(row["is_read"]))
            for row in rows
        )

    def mark_messages_read(
        self,
        team_id: str,
        member_id: str,
        message_ids: tuple[str, ...],
    ) -> None:
        normalized_member = str(UUID(member_id))
        normalized_ids = tuple(dict.fromkeys(str(UUID(item)) for item in message_ids))
        if not normalized_ids:
            return
        with self._transaction() as connection:
            team = self._get_team(connection, team_id)
            self._get_member(connection, team.team_id, normalized_member)
            placeholders = ",".join("?" for _ in normalized_ids)
            rows = connection.execute(
                f"""
                SELECT message_id FROM messages
                WHERE team_id = ? AND recipient_id = ?
                  AND message_id IN ({placeholders})
                """,
                (team.team_id, normalized_member, *normalized_ids),
            ).fetchall()
            if len(rows) != len(normalized_ids):
                raise ValueError("不能标记不存在的邮箱消息。")
            connection.execute(
                f"""
                UPDATE messages SET is_read = 1
                WHERE team_id = ? AND recipient_id = ?
                  AND message_id IN ({placeholders})
                """,
                (team.team_id, normalized_member, *normalized_ids),
            )

    def create_task(self, team_id: str, task: TeamTask) -> TeamTask:
        try:
            with self._transaction() as connection:
                team = self._get_team(connection, team_id)
                tasks = {item.task_id: item for item in self._list_tasks(connection, team.team_id)}
                if task.task_id in tasks:
                    raise ValueError("任务 ID 已存在。")
                self._validate_task_candidate(connection, team.team_id, task, tasks)
                self._insert_task(connection, team.team_id, task)
                return task
        except sqlite3.IntegrityError as error:
            raise ValueError("任务违反 Team 数据约束。") from error

    def list_tasks(self, team_id: str) -> tuple[TeamTask, ...]:
        with self._connection() as connection:
            team = self._get_team(connection, team_id)
            return self._list_tasks(connection, team.team_id)

    def get_task(self, team_id: str, task_id: str) -> TeamTask:
        normalized = str(UUID(task_id))
        with self._connection() as connection:
            team = self._get_team(connection, team_id)
            return self._get_task(connection, team.team_id, normalized)

    def save_task(self, team_id: str, task: TeamTask) -> None:
        try:
            with self._transaction() as connection:
                team = self._get_team(connection, team_id)
                self._save_task(connection, team.team_id, task)
        except sqlite3.IntegrityError as error:
            raise ValueError("任务违反 Team 数据约束。") from error

    def update_task(
        self,
        team_id: str,
        task_id: str,
        updater: Callable[[TeamTask], TeamTask],
    ) -> TeamTask:
        """Atomically update one task from its latest persisted revision."""

        normalized = str(UUID(task_id))
        with self._transaction() as connection:
            team = self._get_team(connection, team_id)
            current = self._get_task(connection, team.team_id, normalized)
            updated = updater(current)
            if updated.task_id != current.task_id:
                raise ValueError("任务更新不能改变任务 ID。")
            if updated != current:
                self._save_task(connection, team.team_id, updated)
            return updated

    def update_task_dependencies(
        self,
        team_id: str,
        task_id: str,
        blocked_by: tuple[str, ...],
    ) -> TeamTask:
        normalized = str(UUID(task_id))
        dependencies = tuple(dict.fromkeys(str(UUID(item)) for item in blocked_by))
        with self._transaction() as connection:
            team = self._get_team(connection, team_id)
            current = self._get_task(connection, team.team_id, normalized)
            candidate = current.revise(blocked_by=dependencies)
            self._save_task(connection, team.team_id, candidate)
            return candidate

    def update_pending_task(
        self,
        team_id: str,
        task_id: str,
        *,
        title: str | None,
        description: str | None,
        blocked_by: tuple[str, ...] | None,
    ) -> TeamTask:
        normalized = str(UUID(task_id))
        with self._transaction() as connection:
            team = self._get_team(connection, team_id)
            current = self._get_task(connection, team.team_id, normalized)
            if current.status is not TaskStatus.PENDING:
                raise ValueError("只有 pending 任务可修改。")
            changes: dict[str, object] = {}
            if blocked_by is not None:
                changes["blocked_by"] = tuple(
                    dict.fromkeys(str(UUID(item)) for item in blocked_by)
                )
            if title is not None:
                changes["title"] = title
            if description is not None:
                changes["description"] = description
            candidate = current.revise(**changes)
            self._save_task(connection, team.team_id, candidate)
            return candidate

    def delete_pending_task(self, team_id: str, task_id: str) -> TeamTask:
        normalized = str(UUID(task_id))
        with self._transaction() as connection:
            team = self._get_team(connection, team_id)
            tasks = {item.task_id: item for item in self._list_tasks(connection, team.team_id)}
            current = tasks.get(normalized)
            if current is None:
                raise ValueError("任务不存在。")
            if current.status is not TaskStatus.PENDING:
                raise ValueError("只有 pending 任务可删除。")
            if any(
                normalized in candidate.blocked_by
                and candidate.status is not TaskStatus.DELETED
                for candidate in tasks.values()
            ):
                raise ValueError("仍有其他任务依赖该任务，不能删除。")
            deleted = current.revise(status=TaskStatus.DELETED)
            self._save_task(connection, team.team_id, deleted)
            return deleted

    def claim_task(self, team_id: str, task_id: str, member_id: str) -> TeamTask:
        normalized_task = str(UUID(task_id))
        normalized_member = str(UUID(member_id))
        with self._transaction() as connection:
            team = self._get_team(connection, team_id)
            self._get_member(connection, team.team_id, normalized_member)
            tasks = {item.task_id: item for item in self._list_tasks(connection, team.team_id)}
            task = tasks.get(normalized_task)
            if task is None:
                raise ValueError("任务不存在。")
            if task.status is not TaskStatus.PENDING or task.assignee_id is not None:
                raise ValueError("任务不可领取。")
            if any(
                tasks[dependency].status is not TaskStatus.COMPLETED
                for dependency in task.blocked_by
            ):
                raise ValueError("任务依赖尚未完成。")
            claimed = task.revise(
                status=TaskStatus.CLAIMED,
                assignee_id=normalized_member,
            )
            self._save_task(connection, team.team_id, claimed)
            return claimed

    def _initialize_schema(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS teams (
                    team_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    data_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS members (
                    team_id TEXT NOT NULL,
                    member_id TEXT NOT NULL,
                    name TEXT NOT NULL COLLATE NOCASE,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (team_id, member_id),
                    UNIQUE (team_id, name),
                    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    team_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    assignee_id TEXT,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (team_id, task_id),
                    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE,
                    FOREIGN KEY (team_id, assignee_id)
                        REFERENCES members(team_id, member_id)
                );

                CREATE TABLE IF NOT EXISTS task_dependencies (
                    team_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    dependency_id TEXT NOT NULL,
                    PRIMARY KEY (team_id, task_id, dependency_id),
                    FOREIGN KEY (team_id, task_id)
                        REFERENCES tasks(team_id, task_id) ON DELETE CASCADE,
                    FOREIGN KEY (team_id, dependency_id)
                        REFERENCES tasks(team_id, task_id)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL UNIQUE,
                    team_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    correlation_id TEXT,
                    created_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    is_read INTEGER NOT NULL DEFAULT 0 CHECK (is_read IN (0, 1)),
                    FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE,
                    FOREIGN KEY (team_id, sender_id)
                        REFERENCES members(team_id, member_id),
                    FOREIGN KEY (team_id, recipient_id)
                        REFERENCES members(team_id, member_id)
                );

                CREATE INDEX IF NOT EXISTS messages_recipient_order
                    ON messages(team_id, recipient_id, sequence);
                CREATE INDEX IF NOT EXISTS messages_correlation
                    ON messages(team_id, correlation_id);

                CREATE TABLE IF NOT EXISTS schema_migrations (
                    migration_key TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                """
            )
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def _import_legacy_teams(self) -> None:
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or _is_link_or_reparse(child):
                continue
            config_path = child / "config.json"
            if not config_path.is_file():
                continue
            try:
                team = TeamRecord.from_dict(_read_json(config_path))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if child.name != team.name:
                continue
            migration_key = f"legacy-json-v1:{team.team_id}"
            with self._connection() as connection:
                imported = connection.execute(
                    "SELECT 1 FROM schema_migrations WHERE migration_key = ?",
                    (migration_key,),
                ).fetchone()
            if imported is not None:
                continue
            members, tasks, messages = self._read_legacy_team(child, team)
            try:
                with self._transaction() as connection:
                    existing_marker = connection.execute(
                        "SELECT 1 FROM schema_migrations WHERE migration_key = ?",
                        (migration_key,),
                    ).fetchone()
                    if existing_marker is not None:
                        continue
                    self._insert_or_verify_team(connection, team)
                    for member in members:
                        self._insert_or_verify_member(connection, team.team_id, member)
                    for task in tasks:
                        self._insert_or_verify_task(connection, team.team_id, task)
                    for task in tasks:
                        self._replace_dependencies(connection, team.team_id, task)
                    for message in messages:
                        self._insert_or_verify_message(connection, team.team_id, message)
                    connection.execute(
                        "INSERT INTO schema_migrations(migration_key, applied_at) VALUES (?, ?)",
                        (migration_key, _utc_now()),
                    )
            except sqlite3.IntegrityError as error:
                raise TeamStoreError(
                    f"旧 Team '{team.name}' 无法完整迁移到 SQLite。"
                ) from error

    def _read_legacy_team(
        self,
        team_dir: Path,
        team: TeamRecord,
    ) -> tuple[tuple[TeamMember, ...], tuple[TeamTask, ...], tuple[TeamMessage, ...]]:
        self._assert_safe_descendant(team_dir)
        members: list[TeamMember] = []
        for path in sorted((team_dir / "members").glob("*.json")):
            member = TeamMember.from_dict(_read_json(path))
            if path.stem != member.member_id:
                raise TeamStoreError("旧成员文件名与成员 ID 不一致。")
            members.append(member)
        tasks: list[TeamTask] = []
        for path in sorted((team_dir / "tasks").glob("*.json")):
            task = TeamTask.from_dict(_read_json(path))
            if path.stem != task.task_id:
                raise TeamStoreError("旧任务文件名与任务 ID 不一致。")
            tasks.append(task)
        member_ids = {member.member_id for member in members}
        messages: list[TeamMessage] = []
        for path in sorted((team_dir / "mailboxes").glob("*.jsonl")):
            if path.stem not in member_ids:
                raise TeamStoreError("旧邮箱不属于已注册成员。")
            if _is_link_or_reparse(path) or path.stat().st_size > _MAX_DOCUMENT_BYTES:
                raise TeamStoreError("旧邮箱文件不安全或过大。")
            mailbox: dict[str, TeamMessage] = {}
            read_ids: set[str] = set()
            for line in path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if not isinstance(event, Mapping):
                    raise TeamStoreError("旧邮箱事件结构无效。")
                if event.get("event") == "message":
                    raw = event.get("message")
                    if not isinstance(raw, Mapping):
                        raise TeamStoreError("旧邮箱消息结构无效。")
                    message = TeamMessage.from_dict(raw)
                    if message.recipient_id != path.stem:
                        raise TeamStoreError("旧邮箱包含其他成员的消息。")
                    mailbox[message.message_id] = message
                elif event.get("event") == "read":
                    raw_ids = event.get("message_ids", ())
                    if not isinstance(raw_ids, list):
                        raise TeamStoreError("旧邮箱已读事件结构无效。")
                    read_ids.update(str(item) for item in raw_ids)
            messages.extend(
                replace(message, read=message_id in read_ids)
                for message_id, message in mailbox.items()
            )
        _assert_acyclic(tasks)
        if any(task.assignee_id not in member_ids for task in tasks if task.assignee_id):
            raise TeamStoreError("旧任务指向不存在的成员。")
        if any(message.sender_id not in member_ids for message in messages):
            raise TeamStoreError("旧消息发送者不属于当前团队。")
        return tuple(members), tuple(tasks), tuple(messages)

    def _build_message(
        self,
        connection: sqlite3.Connection,
        team: TeamRecord,
        actor: ActorContext,
        *,
        to: str,
        body: str,
        summary: str,
        message_type: MessageType,
        payload: Mapping[str, object] | None,
        correlation_id: str | None,
    ) -> TeamMessage:
        sender = self._validate_actor(connection, team.team_id, actor)
        recipient_id = self._resolve_member(connection, team.team_id, to)
        recipient = self._get_member(connection, team.team_id, recipient_id)
        if not body.strip() or len(body) > _MAX_MESSAGE_BODY_CHARS:
            raise ValueError("消息正文为空或过长。")
        if not summary.strip() or len(summary) > _MAX_SUMMARY_CHARS:
            raise ValueError("消息摘要为空或过长。")
        normalized_correlation = None
        if correlation_id is not None:
            normalized_correlation = str(UUID(correlation_id))
        return TeamMessage(
            message_id=str(uuid4()),
            message_type=message_type,
            sender_id=sender.member_id,
            sender_name=sender.name,
            recipient_id=recipient.member_id,
            recipient_name=recipient.name,
            body=body,
            summary=summary,
            created_at=_utc_now(),
            read=False,
            correlation_id=normalized_correlation,
            payload=payload,
        )

    def _insert_message(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        message: TeamMessage,
    ) -> None:
        encoded = _encode(message.to_dict())
        current_bytes = connection.execute(
            """
            SELECT COALESCE(SUM(length(CAST(data_json AS BLOB))), 0)
            FROM messages WHERE team_id = ? AND recipient_id = ?
            """,
            (team_id, message.recipient_id),
        ).fetchone()[0]
        if int(current_bytes) + len(encoded.encode("utf-8")) > _MAX_DOCUMENT_BYTES:
            raise TeamStoreError("邮箱已达到安全上限。")
        connection.execute(
            """
            INSERT INTO messages(
                message_id, team_id, sender_id, recipient_id, correlation_id,
                created_at, data_json, is_read
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.message_id,
                team_id,
                message.sender_id,
                message.recipient_id,
                message.correlation_id,
                message.created_at,
                encoded,
                int(message.read),
            ),
        )

    def _insert_or_verify_message(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        message: TeamMessage,
    ) -> None:
        row = connection.execute(
            "SELECT data_json, is_read FROM messages WHERE message_id = ?",
            (message.message_id,),
        ).fetchone()
        if row is None:
            self._insert_message(connection, team_id, message)
            return
        existing = TeamMessage.from_dict(
            _decode(row["data_json"]), read=bool(row["is_read"])
        )
        if existing != message:
            raise TeamStoreError("旧消息与现有 SQLite 记录冲突。")

    def _insert_member(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        member: TeamMember,
    ) -> None:
        connection.execute(
            """
            INSERT INTO members(team_id, member_id, name, data_json)
            VALUES (?, ?, ?, ?)
            """,
            (team_id, member.member_id, member.name, _encode(member.to_dict())),
        )

    def _insert_or_verify_member(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        member: TeamMember,
    ) -> None:
        row = connection.execute(
            "SELECT data_json FROM members WHERE team_id = ? AND member_id = ?",
            (team_id, member.member_id),
        ).fetchone()
        if row is None:
            self._insert_member(connection, team_id, member)
            return
        if TeamMember.from_dict(_decode(row["data_json"])) != member:
            raise TeamStoreError("旧成员与现有 SQLite 记录冲突。")

    def _save_member(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        member: TeamMember,
    ) -> None:
        self._get_member(connection, team_id, member.member_id)
        cursor = connection.execute(
            """
            UPDATE members SET name = ?, data_json = ?
            WHERE team_id = ? AND member_id = ?
            """,
            (member.name, _encode(member.to_dict()), team_id, member.member_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("成员不存在。")

    def _list_members(
        self,
        connection: sqlite3.Connection,
        team_id: str,
    ) -> tuple[TeamMember, ...]:
        rows = connection.execute(
            """
            SELECT data_json FROM members
            WHERE team_id = ? ORDER BY member_id
            """,
            (team_id,),
        ).fetchall()
        return tuple(TeamMember.from_dict(_decode(row["data_json"])) for row in rows)

    def _get_member(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        member_id: str,
    ) -> TeamMember:
        normalized = str(UUID(member_id))
        row = connection.execute(
            """
            SELECT data_json FROM members
            WHERE team_id = ? AND member_id = ?
            """,
            (team_id, normalized),
        ).fetchone()
        if row is None:
            raise ValueError("成员不属于当前团队。")
        return TeamMember.from_dict(_decode(row["data_json"]))

    def _resolve_member(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        name_or_id: str,
    ) -> str:
        try:
            candidate_id = str(UUID(name_or_id))
        except ValueError:
            candidate_id = ""
        if candidate_id:
            row = connection.execute(
                "SELECT member_id FROM members WHERE team_id = ? AND member_id = ?",
                (team_id, candidate_id),
            ).fetchone()
        else:
            normalized = normalize_team_name(name_or_id)
            row = connection.execute(
                """
                SELECT member_id FROM members
                WHERE team_id = ? AND name = ? COLLATE NOCASE
                """,
                (team_id, normalized),
            ).fetchone()
        if row is None:
            raise ValueError("收件人不是当前团队成员。")
        return str(row["member_id"])

    def _validate_actor(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        actor: ActorContext,
    ) -> TeamMember:
        sender = self._get_member(connection, team_id, actor.member_id)
        if sender.name != actor.member_name:
            raise ValueError("调用方身份与团队注册表不一致。")
        return sender

    def _insert_task(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        task: TeamTask,
    ) -> None:
        connection.execute(
            """
            INSERT INTO tasks(team_id, task_id, assignee_id, data_json)
            VALUES (?, ?, ?, ?)
            """,
            (team_id, task.task_id, task.assignee_id, _encode(task.to_dict())),
        )
        self._replace_dependencies(connection, team_id, task)

    def _insert_or_verify_task(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        task: TeamTask,
    ) -> None:
        row = connection.execute(
            "SELECT data_json FROM tasks WHERE team_id = ? AND task_id = ?",
            (team_id, task.task_id),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO tasks(team_id, task_id, assignee_id, data_json)
                VALUES (?, ?, ?, ?)
                """,
                (team_id, task.task_id, task.assignee_id, _encode(task.to_dict())),
            )
            return
        if TeamTask.from_dict(_decode(row["data_json"])) != task:
            raise TeamStoreError("旧任务与现有 SQLite 记录冲突。")

    def _save_task(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        task: TeamTask,
    ) -> None:
        current = self._get_task(connection, team_id, task.task_id)
        if task.revision <= current.revision:
            raise ValueError("任务 revision 必须递增。")
        tasks = {item.task_id: item for item in self._list_tasks(connection, team_id)}
        self._validate_task_candidate(connection, team_id, task, tasks)
        cursor = connection.execute(
            """
            UPDATE tasks SET assignee_id = ?, data_json = ?
            WHERE team_id = ? AND task_id = ?
            """,
            (task.assignee_id, _encode(task.to_dict()), team_id, task.task_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("任务不存在。")
        self._replace_dependencies(connection, team_id, task)

    def _validate_task_candidate(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        task: TeamTask,
        tasks: Mapping[str, TeamTask],
    ) -> None:
        existing_ids = set(tasks)
        if task.task_id in existing_ids:
            existing_ids.remove(task.task_id)
        available_ids = existing_ids | {task.task_id}
        if task.task_id in task.blocked_by or not set(task.blocked_by).issubset(
            available_ids
        ):
            raise ValueError("任务依赖无效。")
        if task.assignee_id is not None:
            self._get_member(connection, team_id, task.assignee_id)
        candidates = dict(tasks)
        candidates[task.task_id] = task
        _assert_acyclic(candidates.values())

    def _replace_dependencies(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        task: TeamTask,
    ) -> None:
        connection.execute(
            "DELETE FROM task_dependencies WHERE team_id = ? AND task_id = ?",
            (team_id, task.task_id),
        )
        connection.executemany(
            """
            INSERT INTO task_dependencies(team_id, task_id, dependency_id)
            VALUES (?, ?, ?)
            """,
            ((team_id, task.task_id, dependency) for dependency in task.blocked_by),
        )

    def _list_tasks(
        self,
        connection: sqlite3.Connection,
        team_id: str,
    ) -> tuple[TeamTask, ...]:
        rows = connection.execute(
            """
            SELECT data_json FROM tasks
            WHERE team_id = ? ORDER BY task_id
            """,
            (team_id,),
        ).fetchall()
        return tuple(TeamTask.from_dict(_decode(row["data_json"])) for row in rows)

    def _get_task(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        task_id: str,
    ) -> TeamTask:
        normalized = str(UUID(task_id))
        row = connection.execute(
            """
            SELECT data_json FROM tasks
            WHERE team_id = ? AND task_id = ?
            """,
            (team_id, normalized),
        ).fetchone()
        if row is None:
            raise ValueError("任务不存在。")
        return TeamTask.from_dict(_decode(row["data_json"]))

    def _insert_or_verify_team(
        self,
        connection: sqlite3.Connection,
        team: TeamRecord,
    ) -> None:
        row = connection.execute(
            """
            SELECT team_id, data_json FROM teams
            WHERE team_id = ? OR name = ? COLLATE NOCASE
            """,
            (team.team_id, team.name),
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO teams(team_id, name, data_json) VALUES (?, ?, ?)",
                (team.team_id, team.name, _encode(team.to_dict())),
            )
            return
        if str(row["team_id"]) != team.team_id or TeamRecord.from_dict(
            _decode(row["data_json"])
        ) != team:
            raise TeamStoreError("旧团队与现有 SQLite 记录冲突。")

    def _get_team(
        self,
        connection: sqlite3.Connection,
        team_id_or_name: str,
    ) -> TeamRecord:
        row = connection.execute(
            """
            SELECT data_json FROM teams
            WHERE team_id = ? OR name = ? COLLATE NOCASE
            LIMIT 1
            """,
            (team_id_or_name, team_id_or_name.lower()),
        ).fetchone()
        if row is None:
            raise TeamNotFoundError("团队不存在。")
        return TeamRecord.from_dict(_decode(row["data_json"]))

    def _open_connection(self) -> sqlite3.Connection:
        if self.database_path.exists() and _is_link_or_reparse(self.database_path):
            raise TeamStoreError("Team SQLite 数据库不能是链接或 reparse point。")
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._open_connection()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._open_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_root(self) -> None:
        if self.root.exists() and _is_link_or_reparse(self.root):
            raise TeamStoreError("Team 根目录不能是链接或 reparse point。")
        self.root.mkdir(parents=True, exist_ok=True)
        if _is_link_or_reparse(self.root):
            raise TeamStoreError("Team 根目录不能是链接或 reparse point。")

    def _assert_safe_descendant(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.root)
        except ValueError as error:
            raise TeamStoreError("Team 路径越过持久化根目录。") from error
        current = path
        while current != self.root:
            if current.exists() and _is_link_or_reparse(current):
                raise TeamStoreError("Team 路径包含链接或 reparse point。")
            current = current.parent


def _assert_acyclic(tasks: Iterable[TeamTask]) -> None:
    graph = {task.task_id: task.blocked_by for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError("任务依赖不能形成环。")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in graph[task_id]:
            if dependency not in graph:
                raise ValueError("任务依赖不存在。")
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in graph:
        visit(task_id)


def _read_json(path: Path) -> Mapping[str, object]:
    if _is_link_or_reparse(path):
        raise TeamStoreError("Team 状态文件不能是链接或 reparse point。")
    data = path.read_bytes()
    if len(data) > _MAX_DOCUMENT_BYTES:
        raise TeamStoreError("Team 状态文件过大。")
    loaded = json.loads(data.decode("utf-8"))
    if not isinstance(loaded, Mapping):
        raise TeamStoreError("Team 状态文件必须是 JSON 对象。")
    return loaded


def _encode(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_DOCUMENT_BYTES:
        raise TeamStoreError("Team 状态记录过大。")
    return encoded


def _decode(value: str) -> Mapping[str, object]:
    loaded = json.loads(value)
    if not isinstance(loaded, Mapping):
        raise TeamStoreError("Team SQLite 记录必须是 JSON 对象。")
    return loaded


def _is_link_or_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if stat.S_ISLNK(details.st_mode):
        return True
    attributes = getattr(details, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
