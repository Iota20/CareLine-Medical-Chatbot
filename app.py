import os
import time
from functools import lru_cache

import gradio as gr
import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("MEDICAL_MODEL", "medgemma:4b")
TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL", "translategemma:4b")

SYSTEM_PROMPT = (
    "You are a helpful medical information assistant. Provide clear, calm, "
    "evidence-informed general information, explain uncertainty, and suggest "
    "when a user should contact a qualified healthcare professional. Never "
    "claim to diagnose a condition or replace a clinician. For emergencies, "
    "tell the user to contact local emergency services immediately. Keep "
    "responses concise and focused: use short paragraphs or brief bullet points "
    "and include only the most relevant safety information."
)

TRANSLATE_TO_ENGLISH_PROMPT = (
    "Translate the user's Cebuano medical question into clear, natural English "
    "for a medical information model. Preserve important medical terms, drug "
    "names, dosages, symptoms, anatomy, and measurements exactly when possible. "
    "Return only the English translation. Do not answer the question."
)

TRANSLATE_RESPONSE_TO_ENGLISH_PROMPT = (
    "Translate this Cebuano medical assistant response into clear, natural English "
    "for conversation context. Preserve important medical terms, drug names, "
    "dosages, symptoms, anatomy, measurements, warnings, and emergency instructions. "
    "Return only the English translation. Do not add medical advice."
)

TRANSLATE_TO_CEBUANO_PROMPT = (
    "Translate the medical assistant's English response into clear, natural Cebuano. "
    "Preserve important medical terms, drug names, dosages, symptoms, anatomy, "
    "measurements, warnings, and emergency instructions exactly when translation "
    "would reduce precision. Keep the meaning, uncertainty, and safety guidance. "
    "Keep the translation concise and do not expand the response. Return only the "
    "Cebuano translation. Do not add medical advice."
)


@lru_cache(maxsize=2048)
def translate(text, instruction):
    start = time.time()
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": TRANSLATION_MODEL_NAME,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    translated = data.get("message", {}).get("content")
    elapsed = time.time() - start
    print(f"[translate] model={TRANSLATION_MODEL_NAME} instruction={instruction[:40]!r} elapsed={elapsed:.2f}s")
    if not translated:
        raise ValueError("The translation model returned an empty response.")
    return translated.strip()


def _recent_history(history, limit=5):
    if not history:
        return []
    return list(history)[-limit:]


def _text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, (list, tuple)):
        return "".join(_text_content(item) for item in content)
    return ""


def respond(message, history):
    """Send the current conversation to Ollama and return message history."""
    conversation = _recent_history(history or [])
    if not message or not message.strip():
        return conversation

    current_user_message = message.strip()

    def with_reply(reply):
        return conversation + [
            {"role": "user", "content": current_user_message},
            {"role": "assistant", "content": reply},
        ]

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in conversation:
            if isinstance(item, dict):
                role = item.get("role")
                content = _text_content(item.get("content"))
                if role in {"user", "assistant"} and content:
                    instruction = (
                        TRANSLATE_TO_ENGLISH_PROMPT
                        if role == "user"
                        else TRANSLATE_RESPONSE_TO_ENGLISH_PROMPT
                    )
                    messages.append(
                        {"role": role, "content": translate(content, instruction)}
                    )
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                history_user_message, assistant_message = item
                history_user_message = _text_content(history_user_message)
                assistant_message = _text_content(assistant_message)
                if history_user_message:
                    messages.append(
                        {
                            "role": "user",
                            "content": translate(
                                history_user_message, TRANSLATE_TO_ENGLISH_PROMPT
                            ),
                        }
                    )
                if assistant_message:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": translate(
                                assistant_message,
                                TRANSLATE_RESPONSE_TO_ENGLISH_PROMPT,
                            ),
                        }
                    )

        translated_question = translate(current_user_message, TRANSLATE_TO_ENGLISH_PROMPT)
        messages.append({"role": "user", "content": translated_question})
    except requests.exceptions.Timeout:
        return with_reply("The translation took too long. Please try a shorter question.")
    except requests.exceptions.ConnectionError:
        return with_reply(
            "I could not connect to Ollama. Make sure Ollama is running and that "
            f"the `{TRANSLATION_MODEL_NAME}` model is available."
        )
    except requests.exceptions.RequestException as error:
        return with_reply(f"The translation request failed: {error}")
    except (ValueError, KeyError):
        return with_reply("The translation model returned an unexpected response.")

    try:
        start = time.time()
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "messages": messages, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        medical_elapsed = time.time() - start
        print(f"[medical model] model={MODEL_NAME} elapsed={medical_elapsed:.2f}s messages={len(messages)}")
        english_reply = data.get("message", {}).get("content")
        if not english_reply:
            return with_reply("I received an empty response from the medical model.")

        try:
            reply_start = time.time()
            reply = translate(english_reply, TRANSLATE_TO_CEBUANO_PROMPT)
            print(f"[final translation] elapsed={time.time() - reply_start:.2f}s")
        except requests.exceptions.Timeout:
            return with_reply("The translation step took too long. Please try again.")
        return with_reply(reply)
    except requests.exceptions.ConnectionError:
        return with_reply(
            "I could not connect to Ollama. Make sure Ollama is running and "
            f"that the `{MODEL_NAME}` model is available."
        )
    except requests.exceptions.Timeout:
        return with_reply("The request took too long. Please try a shorter question.")
    except requests.exceptions.RequestException as error:
        return with_reply(f"The chatbot request failed: {error}")
    except (ValueError, KeyError):
        return with_reply("Ollama returned an unexpected response.")


def clear_chat():
    return [], None


with gr.Blocks(title="CareLine Medical Chatbot") as demo:
    gr.Markdown(
        """
        # CareLine
        ### A calm place to ask health questions

        Get general medical information and help preparing questions for a
        healthcare professional. This tool is not a diagnosis or a substitute
        for professional medical care.
        """
    )

    gr.Markdown(
        """
        **Before you begin**

        - Do not share passwords, full names, or other private details.
        - For severe or life-threatening symptoms, contact local emergency services.
        - Use the response as general information, not a diagnosis.
        """
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        elem_id="chatbot",
        height=680,
        placeholder="Ask about symptoms, medications, or health topics...",
    )
    with gr.Row():
        message = gr.Textbox(
            placeholder="What would you like to know?",
            label="Your question",
            lines=2,
            scale=8,
        )
        send = gr.Button("Send", variant="primary", scale=1)
    clear = gr.Button("Clear conversation", variant="secondary")

    message.submit(respond, inputs=[message, chatbot], outputs=chatbot).then(
        lambda: "", outputs=message
    )
    send.click(respond, inputs=[message, chatbot], outputs=chatbot).then(
        lambda: "", outputs=message
    )
    clear.click(clear_chat, outputs=[chatbot, message])


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css="""
        #chatbot .message {
            font-size: 0.95rem;
        }
        """,
    )