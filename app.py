import os

import streamlit as st
from ollama import Client


def create_client(api_key: str) -> Client:
    return Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )


def chat_once(client: Client, model: str, prompt: str) -> str:
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return response["message"]["content"]


def fetch_models(client: Client) -> list[str]:
    response = client.list()
    models = []
    for item in response.get("models", []):
        name = item.get("model") or item.get("name")
        if name:
            models.append(name)
    return sorted(set(models))


def main() -> None:
    st.set_page_config(page_title="Ollama Cloud Demo", page_icon=":cloud:")
    st.title("Ollama Cloud LLM Demo")
    st.caption("Run open-source models from Ollama Cloud in a simple Streamlit app.")

    default_key = os.environ.get("OLLAMA_API_KEY", "")
    api_key = st.text_input("OLLAMA_API_KEY", value=default_key, type="password")

    if not api_key:
        st.info("Enter your API key to continue.")
        return

    try:
        client = create_client(api_key)
        models = fetch_models(client)
    except Exception as exc:
        st.error(f"Could not connect/authenticate: {exc}")
        return

    if not models:
        st.warning("No models returned. Check your account or API key.")
        return

    default_candidates = ["gpt-oss:120b", "gpt-oss:20b", "qwen3-coder:480b"]
    default_model = next((m for m in default_candidates if m in models), models[0])
    model = st.selectbox("Model", options=models, index=models.index(default_model))

    prompt = st.text_area(
        "Prompt",
        value="Explain the difference between CPU, GPU, and TPU for AI workloads.",
        height=140,
    )

    if st.button("Send"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return
        with st.spinner("Generating response..."):
            try:
                answer = chat_once(client, model, prompt)
                st.subheader("Response")
                st.write(answer)
            except Exception as exc:
                st.error(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
