import os
import json
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class EmailState(TypedDict):
    email: Dict[str, Any]
    email_category: Optional[str]
    spam_reason: Optional[str]
    is_spam: Optional[bool]
    email_draft: Optional[str]
    messages: List[Dict[str, Any]]


def load_config() -> Dict[str, Any]:
    config_path = os.getenv("APP_CONFIG_FILE")
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).with_name("config.json")

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cfg_value(
    config: Dict[str, Any], section: str, key: str, env_key: str, default: Optional[str] = None
) -> Optional[str]:
    env_val = os.getenv(env_key)
    if env_val:
        return env_val

    section_val = config.get(section, {})
    if isinstance(section_val, dict):
        cfg_val = section_val.get(key)
        if cfg_val:
            return str(cfg_val)
    return default


def cfg_bool(
    config: Dict[str, Any], section: str, key: str, env_key: str, default: bool
) -> bool:
    env_val = os.getenv(env_key)
    if env_val is not None:
        return env_val.strip().lower() in {"1", "true", "yes", "on"}
    section_val = config.get(section, {})
    if isinstance(section_val, dict) and key in section_val:
        val = section_val[key]
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in {"1", "true", "yes", "on"}
    return default


def normalize_openai_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        return base_url
    if path in ("", "/"):
        return base_url.rstrip("/") + "/v1"
    return base_url


def build_model() -> ChatOpenAI:
    config = load_config()
    api_key = cfg_value(config, "openai", "api_key", "OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY. Set env var or put it in langgraph_test/config.json."
        )
    base_url = cfg_value(
        config, "openai", "base_url", "OPENAI_BASE_URL", "https://codeflow.asia/v1"
    )
    if not base_url:
        raise ValueError("Missing OPENAI_BASE_URL.")
    normalized_base_url = normalize_openai_base_url(base_url)
    if normalized_base_url != base_url:
        print(f"Adjusted OPENAI_BASE_URL to {normalized_base_url}")

    timeout_s = float(
        cfg_value(config, "openai", "timeout_seconds", "OPENAI_TIMEOUT_SECONDS", "60") or 60
    )
    trust_env = cfg_bool(config, "openai", "trust_env_proxy", "OPENAI_TRUST_ENV_PROXY", False)
    http_client = httpx.Client(timeout=timeout_s, trust_env=trust_env)

    return ChatOpenAI(
        model=cfg_value(config, "openai", "model", "OPENAI_MODEL", "gpt-5.2"),
        openai_api_key=api_key,
        base_url=normalized_base_url,
        max_retries=3,
        http_client=http_client,
        temperature=0,
    )


model = build_model()


def to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def read_email(state: EmailState) -> Dict[str, Any]:
    email = state["email"]
    print(
        f"Alfred is processing an email from {email['sender']} with subject: "
        f"{email['subject']}"
    )
    return {}


def classify_email(state: EmailState) -> Dict[str, Any]:
    email = state["email"]
    prompt = f"""
As Alfred the butler, analyze this email.

Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

Return exactly three lines:
label: spam or legitimate
reason: <short reason>
category: <inquiry/complaint/thank you/request/information or none>
"""
    response = model.invoke([HumanMessage(content=prompt)])
    response_text = to_text(response.content)
    response_lower = response_text.lower()

    is_spam = "label: spam" in response_lower
    if not is_spam:
        is_spam = "spam" in response_lower and "not spam" not in response_lower

    spam_reason = None
    email_category = None
    for line in response_text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("reason:"):
            value = stripped.split(":", 1)[1].strip()
            spam_reason = value or None
        if lowered.startswith("category:"):
            value = stripped.split(":", 1)[1].strip().lower()
            if value and value != "none":
                email_category = value

    if not is_spam and email_category is None:
        for category in ["inquiry", "complaint", "thank you", "request", "information"]:
            if category in response_lower:
                email_category = category
                break

    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response_text},
    ]
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages,
    }


def route_email(state: EmailState) -> str:
    return "spam" if state.get("is_spam") else "legitimate"


def handle_spam(state: EmailState) -> Dict[str, Any]:
    print(f"Alfred has marked the email as spam. Reason: {state.get('spam_reason')}")
    print("The email has been moved to the spam folder.")
    return {}


def draft_response(state: EmailState) -> Dict[str, Any]:
    email = state["email"]
    category = state.get("email_category") or "general"
    prompt = f"""
As Alfred the butler, draft a polite preliminary response to this email.

Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

This email has been categorized as: {category}
Draft a brief, professional response that Mr. Hugg can review and personalize.
"""
    response = model.invoke([HumanMessage(content=prompt)])
    response_text = to_text(response.content)
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response_text},
    ]
    return {"email_draft": response_text, "messages": new_messages}


def notify_mr_hugg(state: EmailState) -> Dict[str, Any]:
    email = state["email"]
    print("\n" + "=" * 50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state.get('email_category')}")
    print("\nI've prepared a draft response for your review:")
    print("-" * 50)
    print(state.get("email_draft"))
    print("=" * 50 + "\n")
    return {}


def get_langfuse_callbacks() -> Tuple[Optional[List[Any]], Optional[Any]]:
    config = load_config()
    public_key = cfg_value(config, "langfuse", "public_key", "LANGFUSE_PUBLIC_KEY")
    secret_key = cfg_value(config, "langfuse", "secret_key", "LANGFUSE_SECRET_KEY")
    host = cfg_value(
        config, "langfuse", "host", "LANGFUSE_HOST", "https://us.cloud.langfuse.com"
    )
    if not public_key or not secret_key:
        return None, None

    # Some providers/docs use pk/sk prefixes; auto-correct a common swap mistake.
    if public_key.startswith("sk-") and secret_key.startswith("pk-"):
        public_key, secret_key = secret_key, public_key
        print("Detected swapped Langfuse keys; auto-corrected public/secret assignment.")

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
    os.environ.setdefault("LANGFUSE_HOST", host)

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:
        print(
            "LANGFUSE keys found, but langfuse-langchain integration deps are missing "
            "(install `langchain`). Skip tracing."
        )
        return None, None

    timeout_s = float(
        cfg_value(
            config, "langfuse", "timeout_seconds", "LANGFUSE_TIMEOUT_SECONDS", "10"
        )
        or 10
    )
    trust_env = cfg_bool(
        config, "langfuse", "trust_env_proxy", "LANGFUSE_TRUST_ENV_PROXY", False
    )

    # Initialize an explicit Langfuse client so we control proxy/timeout behavior.
    langfuse_http_client = httpx.Client(timeout=timeout_s, trust_env=trust_env)
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        httpx_client=langfuse_http_client,
    )

    if hasattr(client, "auth_check"):
        try:
            if not client.auth_check():
                print("Langfuse auth_check failed. Skip tracing.")
                return None, None
        except Exception as exc:
            # For network/proxy blips, keep tracing enabled and let async exporter retry.
            if type(exc).__name__ == "UnauthorizedError":
                print(f"Langfuse credentials/host invalid ({exc}). Skip tracing.")
                return None, None
            print(f"Langfuse auth check transient error ({exc}). Continue with tracing.")

    handler = CallbackHandler(public_key=public_key)
    return [handler], client


def initial_state(email: Dict[str, str]) -> EmailState:
    return {
        "email": email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "messages": [],
    }


email_graph = StateGraph(EmailState)
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

email_graph.add_edge(START, "read_email")
email_graph.add_edge("read_email", "classify_email")
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {"spam": "handle_spam", "legitimate": "draft_response"},
)
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

compiled_graph = email_graph.compile()

legitimate_email = {
    "sender": "candyai@example.com",
    "subject": "We have special services for you to relax",
    "body": (
        "Dear Mr. Hugg, you are so lucky to have the opportunity to try our special services. "
        "Please let us know if you are interested!"
    ),
}
spam_email = {
    "sender": "loser@doubaointl.com",
    "subject": "You lose! Claim your prize now!!!",
    "body": (
        "You lose! Claim your prize now!!! Click this link to win a million dollars: http://nongyeyinhang.link/win?user=Hugg"
    ),
}

callbacks, langfuse_client = get_langfuse_callbacks()
invoke_config = {"callbacks": callbacks} if callbacks else None

try:
    print("\nProcessing legitimate email...")
    if invoke_config:
        legitimate_result = compiled_graph.invoke(
            input=initial_state(legitimate_email), config=invoke_config
        )
    else:
        legitimate_result = compiled_graph.invoke(initial_state(legitimate_email))

    print("\nProcessing spam email...")
    if invoke_config:
        spam_result = compiled_graph.invoke(
            input=initial_state(spam_email), config=invoke_config
        )
    else:
        spam_result = compiled_graph.invoke(initial_state(spam_email))
finally:
    if langfuse_client:
        try:
            langfuse_client.flush()
            langfuse_client.shutdown()
        except Exception as exc:
            print(f"Langfuse flush/shutdown skipped due to error: {exc}")
