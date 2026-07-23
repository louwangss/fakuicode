"""Safe, user-facing error types."""

from typing import Literal


ProviderErrorCategory = Literal["other", "transient", "context_overflow"]
ProviderFailurePhase = Literal[
    "request",
    "http_status",
    "stream_event",
    "stream_transport",
    "stream_format",
]
ProviderErrorType = Literal[
    "invalid_request_error",
    "authentication_error",
    "billing_error",
    "permission_error",
    "not_found_error",
    "request_too_large",
    "rate_limit_error",
    "api_error",
    "overloaded_error",
    "unknown_error",
]
PROVIDER_FAILURE_PHASE_VALUES = frozenset(
    {"request", "http_status", "stream_event", "stream_transport", "stream_format"}
)
PROVIDER_ERROR_TYPE_VALUES = frozenset(
    {
        "invalid_request_error",
        "authentication_error",
        "billing_error",
        "permission_error",
        "not_found_error",
        "request_too_large",
        "rate_limit_error",
        "api_error",
        "overloaded_error",
        "unknown_error",
    }
)


def normalize_provider_request_id(value: object) -> str | None:
    """Accept only a bounded identifier that is safe to render in one terminal line."""

    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        return None
    if not all(
        character.isascii() and (character.isalnum() or character in "._:-")
        for character in value
    ):
        return None
    return value


class ConfigurationError(ValueError):
    """Raised when a configuration cannot be used safely."""


class PermissionConfigurationError(ConfigurationError):
    """Raised when a permission configuration source is invalid."""


class HookConfigurationError(ConfigurationError):
    """Raised when a lifecycle Hook configuration source is invalid."""


class PermissionPersistenceError(RuntimeError):
    """Raised when a permission or trust choice cannot be saved safely."""


class ProviderError(RuntimeError):
    """Raised when a provider request or stream fails."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        category: ProviderErrorCategory = "other",
        status_code: int | None = None,
        error_type: ProviderErrorType | None = None,
        failure_phase: ProviderFailurePhase | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category: ProviderErrorCategory = (
            "transient" if retryable and category == "other" else category
        )
        self.retryable = retryable or category == "transient"
        self.status_code = (
            status_code if isinstance(status_code, int) and 100 <= status_code <= 599 else None
        )
        self.error_type = error_type if error_type in PROVIDER_ERROR_TYPE_VALUES else None
        self.failure_phase = (
            failure_phase if failure_phase in PROVIDER_FAILURE_PHASE_VALUES else None
        )
        self.request_id = normalize_provider_request_id(request_id)


class ProviderCapabilityError(ProviderError):
    """Raised before a Provider call when required system channels are unavailable."""

    def __init__(self) -> None:
        super().__init__("Provider cannot safely accept system instructions.")


class RequestCancelled(RuntimeError):
    """Raised when the user cancels an active model request."""

    def __init__(self) -> None:
        super().__init__("Request cancelled.")


class ToolPolicyError(RuntimeError):
    """Raised when a requested tool action violates a local safety boundary."""


class ToolExecutionError(RuntimeError):
    """Raised when an allowed local tool cannot complete its action."""
