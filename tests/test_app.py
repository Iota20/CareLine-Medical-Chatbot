import app


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": self._content}}


def test_translate_uses_cache(monkeypatch):
    app.translate.cache_clear()
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json)
        content = json["messages"][-1]["content"]
        return FakeResponse(f"translated:{content}")

    monkeypatch.setattr(app.requests, "post", fake_post)

    first = app.translate("migula ko", app.TRANSLATE_TO_ENGLISH_PROMPT)
    second = app.translate("migula ko", app.TRANSLATE_TO_ENGLISH_PROMPT)

    assert first == second == "translated:migula ko"
    assert len(calls) == 1


def test_respond_uses_recent_history_only(monkeypatch):
    app.translate.cache_clear()
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json)
        model = json["model"]
        content = json["messages"][-1]["content"]
        if model == app.TRANSLATION_MODEL_NAME:
            return FakeResponse(f"translated:{content}")
        return FakeResponse(f"medical:{content}")

    monkeypatch.setattr(app.requests, "post", fake_post)
    history = [[f"user_{i}", f"assistant_{i}"] for i in range(20)]

    result = app.respond("new question", history)
    translation_calls = sum(
        1 for payload in calls if payload["model"] == app.TRANSLATION_MODEL_NAME
    )

    assert translation_calls <= 13
    assert len(result) == 7
    assert result[-1]["role"] == "assistant"


def test_respond_handles_list_message_content(monkeypatch):
    app.translate.cache_clear()

    def fake_post(url, json, timeout):
        model = json["model"]
        content = json["messages"][-1]["content"]
        if model == app.TRANSLATION_MODEL_NAME:
            return FakeResponse(f"translated:{content}")
        return FakeResponse(f"medical:{content}")

    monkeypatch.setattr(app.requests, "post", fake_post)
    history = [{"role": "user", "content": [{"text": "previous question"}]}]

    result = app.respond("new question", history)

    assert result[-1]["role"] == "assistant"
