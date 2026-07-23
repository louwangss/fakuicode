from __future__ import annotations

from threading import Event

import httpx
import pytest

from fakuicode.errors import RequestCancelled
from fakuicode.instructions.models import DEFAULT_INSTRUCTION_LIMITS
from fakuicode.skills.install import (
    GitHubSkillFetcher,
    SkillInstallError,
    parse_install_source,
)


def test_skills_sh_url_resolves_to_public_github_source_and_skill() -> None:
    source = parse_install_source(
        "https://www.skills.sh/anthropics/skills/frontend-design"
    )

    assert source.owner == "anthropics"
    assert source.repo == "skills"
    assert source.requested_skill == "frontend-design"
    assert source.canonical_url == "https://github.com/anthropics/skills"
    assert source.ref is None
    assert source.skill_path is None


def test_github_tree_url_preserves_explicit_ref_and_skill_path() -> None:
    source = parse_install_source(
        "https://github.com/anthropics/skills/tree/main/skills/frontend-design"
    )

    assert source.ref == "main"
    assert source.skill_path == "skills/frontend-design"
    assert source.requested_skill == "frontend-design"


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/anthropics/skills",
        "https://gitlab.com/anthropics/skills",
        "https://github.com@127.0.0.1/anthropics/skills",
        "https://skills.sh/mintlify.com/mintlify/frontend",
        "https://github.com/anthropics/skills/issues/1",
        "https://github.com/acme/skills/tree/main/demo.",
        "https://www.skills.sh/acme/skills/con",
    ],
)
def test_unsupported_or_unsafe_sources_are_rejected(url: str) -> None:
    with pytest.raises(SkillInstallError):
        parse_install_source(url)


def test_fetcher_pins_commit_and_downloads_only_selected_skill_files() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path == "/repos/anthropics/skills":
            return httpx.Response(200, json={"default_branch": "main"})
        if request.url.path == "/repos/anthropics/skills/commits/main":
            return httpx.Response(
                200,
                json={"sha": "a" * 40, "commit": {"tree": {"sha": "b" * 40}}},
            )
        if request.url.path == f"/repos/anthropics/skills/git/trees/{'b' * 40}":
            return httpx.Response(
                200,
                json={
                    "truncated": False,
                    "tree": [
                        {"path": "README.md", "type": "blob", "mode": "100644", "size": 20},
                        {
                            "path": "skills/frontend-design/SKILL.md",
                            "type": "blob",
                            "mode": "100644",
                            "size": 120,
                        },
                        {
                            "path": "skills/frontend-design/LICENSE.txt",
                            "type": "blob",
                            "mode": "100644",
                            "size": 20,
                        },
                        {
                            "path": "skills/other/SKILL.md",
                            "type": "blob",
                            "mode": "100644",
                            "size": 80,
                        },
                    ],
                },
            )
        if request.url.host == "raw.githubusercontent.com" and request.url.path.endswith(
            "/skills/frontend-design/SKILL.md"
        ):
            return httpx.Response(
                200,
                content=(
                    b"---\nname: frontend-design\n"
                    b"description: Build frontends\nlicense: Complete terms in LICENSE.txt\n"
                    b"---\nBuild it.\n"
                ),
            )
        if request.url.host == "raw.githubusercontent.com" and request.url.path.endswith(
            "/skills/frontend-design/LICENSE.txt"
        ):
            return httpx.Response(200, content=b"license\n")
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    package = GitHubSkillFetcher(client=client).fetch(
        parse_install_source("https://www.skills.sh/anthropics/skills/frontend-design")
    )

    assert package.name == "frontend-design"
    assert package.revision == "a" * 40
    assert package.skill_path == "skills/frontend-design"
    assert set(package.files) == {"SKILL.md", "LICENSE.txt"}
    assert not any("skills/other" in request for request in requests)
    assert all("/main/" not in request for request in requests if "raw.githubusercontent.com" in request)


def test_repository_with_multiple_skills_requires_an_explicit_selection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/acme/skills":
            return httpx.Response(200, json={"default_branch": "main"})
        if request.url.path == "/repos/acme/skills/commits/main":
            return httpx.Response(
                200,
                json={"sha": "a" * 40, "commit": {"tree": {"sha": "b" * 40}}},
            )
        return httpx.Response(
            200,
            json={
                "truncated": False,
                "tree": [
                    {"path": "skills/one/SKILL.md", "type": "blob", "mode": "100644", "size": 1},
                    {"path": "skills/two/SKILL.md", "type": "blob", "mode": "100644", "size": 1},
                ],
            },
        )

    fetcher = GitHubSkillFetcher(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(SkillInstallError, match="one.*two"):
        fetcher.fetch(parse_install_source("https://github.com/acme/skills"))


def test_github_repository_url_can_install_a_root_skill() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/acme/root-skill":
            return httpx.Response(200, json={"default_branch": "main"})
        if request.url.path == "/repos/acme/root-skill/commits/main":
            return httpx.Response(
                200,
                json={"sha": "a" * 40, "commit": {"tree": {"sha": "b" * 40}}},
            )
        if request.url.path == f"/repos/acme/root-skill/git/trees/{'b' * 40}":
            return httpx.Response(
                200,
                json={
                    "truncated": False,
                    "tree": [
                        {"path": "SKILL.md", "type": "blob", "mode": "100644", "size": 80},
                        {"path": "LICENSE", "type": "blob", "mode": "100644", "size": 10},
                    ],
                },
            )
        if request.url.path.endswith("/SKILL.md"):
            return httpx.Response(
                200,
                content=b"---\nname: root-skill\ndescription: Root package\n---\nUse it.\n",
            )
        return httpx.Response(200, content=b"license\n")

    package = GitHubSkillFetcher(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    ).fetch(parse_install_source("https://github.com/acme/root-skill"))

    assert package.name == "root-skill"
    assert package.skill_path == "."
    assert set(package.files) == {"SKILL.md", "LICENSE"}


@pytest.mark.parametrize("unsafe_type,mode", [("blob", "120000"), ("commit", "160000")])
def test_selected_package_rejects_symlinks_and_submodules(unsafe_type: str, mode: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/acme/skills":
            return httpx.Response(200, json={"default_branch": "main"})
        if request.url.path == "/repos/acme/skills/commits/main":
            return httpx.Response(
                200,
                json={"sha": "a" * 40, "commit": {"tree": {"sha": "b" * 40}}},
            )
        return httpx.Response(
            200,
            json={
                "truncated": False,
                "tree": [
                    {"path": "demo/SKILL.md", "type": "blob", "mode": "100644", "size": 10},
                    {"path": "demo/unsafe", "type": unsafe_type, "mode": mode, "size": 3},
                ],
            },
        )

    fetcher = GitHubSkillFetcher(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(SkillInstallError, match="unsafe"):
        fetcher.fetch(parse_install_source("https://github.com/acme/skills", skill="demo"))


def test_fetch_cancellation_is_propagated_before_network_access() -> None:
    cancel = Event()
    cancel.set()
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("network should not be used"))
    )

    with pytest.raises(RequestCancelled):
        GitHubSkillFetcher(client=client).fetch(
            parse_install_source("https://github.com/acme/skills", skill="demo"),
            cancel_event=cancel,
        )


@pytest.mark.parametrize(
    "response,error",
    [
        (httpx.Response(302, headers={"location": "https://example.test"}), "redirected"),
        (httpx.Response(429, headers={"retry-after": "60"}), "rate limit"),
    ],
)
def test_fetcher_rejects_redirects_and_rate_limits(response: httpx.Response, error: str) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: response))

    with pytest.raises(SkillInstallError, match=error):
        GitHubSkillFetcher(client=client).fetch(
            parse_install_source("https://github.com/acme/skills", skill="demo")
        )


def test_fetcher_rejects_truncated_repository_tree() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/acme/skills":
            return httpx.Response(200, json={"default_branch": "main"})
        if request.url.path == "/repos/acme/skills/commits/main":
            return httpx.Response(
                200,
                json={"sha": "a" * 40, "commit": {"tree": {"sha": "b" * 40}}},
            )
        return httpx.Response(200, json={"truncated": True, "tree": []})

    with pytest.raises(SkillInstallError, match="incomplete"):
        GitHubSkillFetcher(client=httpx.Client(transport=httpx.MockTransport(handler))).fetch(
            parse_install_source("https://github.com/acme/skills", skill="demo")
        )


def test_fetcher_maps_network_timeout_to_a_bounded_install_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(SkillInstallError, match="request failed"):
        GitHubSkillFetcher(client=httpx.Client(transport=httpx.MockTransport(handler))).fetch(
            parse_install_source("https://github.com/acme/skills", skill="demo")
        )


def test_fetcher_stops_before_downloading_a_declared_oversized_file() -> None:
    raw_requested = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal raw_requested
        if request.url.path == "/repos/acme/skills":
            return httpx.Response(200, json={"default_branch": "main"})
        if request.url.path == "/repos/acme/skills/commits/main":
            return httpx.Response(
                200,
                json={"sha": "a" * 40, "commit": {"tree": {"sha": "b" * 40}}},
            )
        if request.url.host == "raw.githubusercontent.com":
            raw_requested = True
            return httpx.Response(200, content=b"unexpected")
        return httpx.Response(
            200,
            json={
                "truncated": False,
                "tree": [
                    {
                        "path": "demo/SKILL.md",
                        "type": "blob",
                        "mode": "100644",
                        "size": DEFAULT_INSTRUCTION_LIMITS.max_source_bytes + 1,
                    }
                ],
            },
        )

    with pytest.raises(SkillInstallError, match="too large"):
        GitHubSkillFetcher(client=httpx.Client(transport=httpx.MockTransport(handler))).fetch(
            parse_install_source("https://github.com/acme/skills", skill="demo")
        )
    assert raw_requested is False


def test_fetcher_rejects_too_many_files_before_downloading() -> None:
    raw_requested = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal raw_requested
        if request.url.path == "/repos/acme/skills":
            return httpx.Response(200, json={"default_branch": "main"})
        if request.url.path == "/repos/acme/skills/commits/main":
            return httpx.Response(
                200,
                json={"sha": "a" * 40, "commit": {"tree": {"sha": "b" * 40}}},
            )
        if request.url.host == "raw.githubusercontent.com":
            raw_requested = True
            return httpx.Response(200, content=b"unexpected")
        files = [
            {"path": "demo/SKILL.md", "type": "blob", "mode": "100644", "size": 10}
        ]
        files.extend(
            {
                "path": f"demo/assets/{index}.txt",
                "type": "blob",
                "mode": "100644",
                "size": 1,
            }
            for index in range(DEFAULT_INSTRUCTION_LIMITS.max_file_targets - 1)
        )
        return httpx.Response(200, json={"truncated": False, "tree": files})

    with pytest.raises(SkillInstallError, match="too many"):
        GitHubSkillFetcher(client=httpx.Client(transport=httpx.MockTransport(handler))).fetch(
            parse_install_source("https://github.com/acme/skills", skill="demo")
        )
    assert raw_requested is False


def test_fetcher_rejects_case_colliding_and_reserved_receipt_paths() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/acme/skills":
            return httpx.Response(200, json={"default_branch": "main"})
        if request.url.path == "/repos/acme/skills/commits/main":
            return httpx.Response(
                200,
                json={"sha": "a" * 40, "commit": {"tree": {"sha": "b" * 40}}},
            )
        return httpx.Response(
            200,
            json={
                "truncated": False,
                "tree": [
                    {"path": "demo/SKILL.md", "type": "blob", "mode": "100644", "size": 10},
                    {"path": "demo/skill.md", "type": "blob", "mode": "100644", "size": 10},
                    {
                        "path": "demo/.FAKUICODE/INSTALL.YAML",
                        "type": "blob",
                        "mode": "100644",
                        "size": 10,
                    },
                ],
            },
        )

    with pytest.raises(SkillInstallError, match="collision|reserved"):
        GitHubSkillFetcher(client=httpx.Client(transport=httpx.MockTransport(handler))).fetch(
            parse_install_source("https://github.com/acme/skills", skill="demo")
        )
