# Ollama Cloud Tutorial: Use Open Source LLMs Without Downloading Large Models

This tutorial is designed for a 1-hour YouTube walkthrough for **Murat Karakaya Akademi**.
It is beginner-friendly and fully step-by-step.

You will learn how to:
- use open-source LLMs via Ollama Cloud without needing a powerful local GPU
- test multiple models without downloading all of them to your disk
- run models from CLI
- call Ollama Cloud API with Python
- use two independent Python samples (`compare_models.py` and `app.py`)

## 1. What Is Ollama Cloud?

Ollama Cloud lets you run supported cloud models while keeping a familiar Ollama workflow.
This is useful when a model is too large for your GPU VRAM or when you want to evaluate multiple large models quickly.

In practice, you get two usage styles:
- local CLI flow with cloud models (for example `gpt-oss:120b-cloud`) ([Cloud docs](https://docs.ollama.com/cloud))
- direct cloud API flow via `https://ollama.com/api` and an API key ([API docs](https://docs.ollama.com/api), [Authentication](https://docs.ollama.com/api/authentication))

What you achieved in this step: You now understand what Ollama Cloud is and the two main ways to use it.

## 2. Why Use Ollama Cloud?

Use Ollama Cloud when:
- your available GPU VRAM cannot fit larger models
- you do not want to download many large model files to local storage
- you want to test and compare several open models quickly
- you want to keep your local development environment simple

Pricing note:
- Ollama Cloud is often used to start and test quickly, but always check current policy and limits on the [official cloud page](https://docs.ollama.com/cloud).

What you achieved in this step: You now know practical reasons to prefer cloud-based model testing.

## 3. When to Use Ollama Cloud

Choose Ollama Cloud for:
- rapid prototyping
- model benchmarking and comparison
- educational demos/tutorials
- environments where hardware is limited

Choose local-only models when:
- you need offline execution
- your policy requires fully local inference
- you have enough hardware and want full local control

What you achieved in this step: You can decide when cloud-first vs local-first is better.

## 4. Prerequisites

- Windows 10/11
- internet connection
- Ollama account
- Python 3.8+
- VS Code (Python extension)

Quick checks in PowerShell:

```powershell
python --version
ollama --version
```

If `ollama` is not recognized, install it in the next step and reopen PowerShell.

Helpful docs:
- [Windows install](https://docs.ollama.com/windows)
- [Ollama Cloud overview](https://docs.ollama.com/cloud)

What you achieved in this step: You verified the tools needed before running cloud models and Python code.

## 5. Ollama Cloud in Action

### 5.1. Get an API Key

1. Open your Ollama account settings.
2. Create an API key.
3. Copy it and store it securely.

Reference:
- [Cloud docs](https://docs.ollama.com/cloud)
- [API authentication](https://docs.ollama.com/api/authentication)

Set API key for current PowerShell session:

```powershell
$env:OLLAMA_API_KEY = "your_api_key_here"
```

Set API key persistently for your user (new terminal required):

```powershell
setx OLLAMA_API_KEY "your_api_key_here"
```

Verify in current session:

```powershell
echo $env:OLLAMA_API_KEY
```

What you achieved in this step: You configured authentication for direct Ollama Cloud API usage.

### 5.2. Download and Install Ollama on PC

Use the official installer from [Ollama Windows docs](https://docs.ollama.com/windows).
After installation, open a new PowerShell window and check:

```powershell
ollama --version
```

Optional custom install path example:

```powershell
OllamaSetup.exe /DIR="D:\Ollama"
```

What you achieved in this step: You installed Ollama and confirmed CLI access.

### 5.3. Login to Ollama Cloud from CLI

Sign in once:

```powershell
ollama signin
```

This enables local Ollama to authenticate cloud-required operations.

Reference:
- [Ollama Cloud docs](https://docs.ollama.com/cloud)

What you achieved in this step: Your local CLI can now use cloud models via signed-in session.

### 5.4. Use Ollama CLI with Cloud Models

Pull and run a cloud model:

```powershell
ollama pull gpt-oss:120b-cloud
ollama run gpt-oss:120b-cloud
```

Inside interactive prompt, try:

```text
Explain transformer architecture in simple terms.
```

Exit interactive mode:

```text
/bye
```

Cloud naming note:
- CLI cloud usage commonly uses the `-cloud` suffix.
- direct cloud API model IDs may differ from CLI naming.

Reference:
- [Cloud model usage](https://docs.ollama.com/cloud)

What you achieved in this step: You ran a large cloud model from CLI without local large-model download.

### 5.5. Access Ollama Cloud API Directly

For direct API access, use `https://ollama.com/api` with `Authorization: Bearer <OLLAMA_API_KEY>` ([Authentication docs](https://docs.ollama.com/api/authentication)).

List models:

```powershell
curl https://ollama.com/api/tags `
  -H "Authorization: Bearer $env:OLLAMA_API_KEY"
```

Endpoint reference: [List models (`/api/tags`)](https://docs.ollama.com/api/tags)

Send a chat request:

```powershell
curl https://ollama.com/api/chat `
  -H "Authorization: Bearer $env:OLLAMA_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{
    "model": "gpt-oss:120b",
    "messages": [
      {"role": "user", "content": "Why is the sky blue?"}
    ],
    "stream": false
  }'
```

Endpoint reference: [Chat (`/api/chat`)](https://docs.ollama.com/api/chat)

What you achieved in this step: You called Ollama Cloud directly over HTTPS using API-key authentication.

### 5.6. Code Samples Overview

Sample 1: [`compare_models.py`](compare_models.py)
- Goal: benchmark multiple cloud models on structured tasks
- Type: terminal/CLI script
- Dependency relation: independent sample (does not call `app.py`)

Sample 2: [`app.py`](app.py)
- Goal: run a Streamlit web chat app against Ollama Cloud
- Type: web app sample
- Dependency relation: independent sample (does not call `compare_models.py`)

Shared requirements:
- both samples use `OLLAMA_API_KEY`
- install dependencies from [`requirements.txt`](requirements.txt)

Quick code review commands:

```powershell
type compare_models.py
type app.py
```

Python reference note:
- this client pattern follows [Ollama Cloud Python docs](https://docs.ollama.com/cloud#python): `Client(host="https://ollama.com", headers={"Authorization": "Bearer ..."})`

Repository Python files map:
- [`compare_models.py`](compare_models.py): model benchmarking script
- [`app.py`](app.py): Streamlit chat application
- [`requirements.txt`](requirements.txt): dependencies

What you achieved in this step: You understand how the two sample files are scoped and related.

### 5.7. Sample 1: Compare 3 Open Models Quickly

#### 5.7.1. Overview

File: [`compare_models.py`](compare_models.py)

- Goal: benchmark multiple cloud models on structured tasks
- Input: `OLLAMA_API_KEY` and model list inside the script
- Output: console summary tables and `model_comparison_report.md`
- Best use case: choose a baseline model before building an app

#### 5.7.2. Setup

Install dependencies from repo root:

```powershell
cd C:\Codes\OllamaCloud
pip install -r requirements.txt
```

Ensure your API key is available:

```powershell
$env:OLLAMA_API_KEY = "your_api_key_here"
```

#### 5.7.3. Code Notes

In `compare_models.py`, key parts are:
- `MODELS`: model IDs to compare
- `TASKS`: structured prompts for evaluation
- `quality_score(...)`: keyword/format-based score
- report writer: saves `model_comparison_report.md`

#### 5.7.4. Run

```powershell
python compare_models.py
```

Tip:
- if one model is unavailable, list current models with `/api/tags` and replace it.

#### 5.7.5. References

- [Cloud Python usage](https://docs.ollama.com/cloud#python)
- [ollama-python README](https://github.com/ollama/ollama-python)

What you achieved in this step: You built a repeatable model-comparison workflow for cloud models.

### 5.8. Sample 2: Build a Simple Python Web App in VS Code (Streamlit)

#### 5.8.1. Overview

File: [`app.py`](app.py)

- Goal: provide a minimal Streamlit chat app for Ollama Cloud
- Input: `OLLAMA_API_KEY` and user prompt
- Output: browser chat UI at `http://localhost:8501`
- Best use case: interactive demo and manual prompt testing

#### 5.8.2. Setup

From the repository root (`C:\Codes\OllamaCloud`), create and activate a virtual environment:

```powershell
cd C:\Codes\OllamaCloud
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5.8.3. Code Notes

Use the ready app in this repo: [`app.py`](app.py)

Main logic in `app.py`:
- `create_client(api_key)`: creates `ollama.Client` with cloud host and Bearer header
- `fetch_models(client)`: fetches available models from cloud
- `chat_once(client, model, prompt)`: sends one chat request and returns text
- `main()`: Streamlit flow (API key input, model selection, prompt, response display)

#### 5.8.4. Run

```powershell
streamlit run app.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`), enter API key, select model, and chat.

#### 5.8.5. References

- [Cloud Python usage](https://docs.ollama.com/cloud#python)
- [ollama-python package usage](https://github.com/ollama/ollama-python)

What you achieved in this step: You built and ran an interactive cloud-based LLM web app.

## 6. Common Errors and Fixes

`401 Unauthorized`:
- API key is missing or invalid
- ensure `OLLAMA_API_KEY` is set in current shell
- verify Bearer header format
- check [Authentication docs](https://docs.ollama.com/api/authentication)

`404 model not found`:
- selected model is not available for your account
- run `/api/tags` and pick a model from returned list
- endpoint docs: [List models](https://docs.ollama.com/api/tags)

CLI fails to use cloud model:
- confirm `ollama signin` is completed
- retry exact cloud model ID (for example `gpt-oss:120b-cloud`)

PowerShell variable not found:
- if key was set with `setx`, open a new terminal
- or set current session with `$env:OLLAMA_API_KEY = "..."`

Network/timeouts:
- retry request
- try a smaller/less busy model
- verify firewall/proxy rules

What you achieved in this step: You can diagnose and resolve common runtime issues quickly.

## 7. Conclusion

You now have a full workflow to:
- run cloud models using Ollama CLI
- call Ollama Cloud API directly
- compare multiple models rapidly
- run a simple Python Streamlit app in VS Code

This approach is suitable for learning, prototyping, and model evaluation without local large-model downloads.

What you achieved in this step: You completed an end-to-end Ollama Cloud tutorial ready for YouTube production.

## 8. Repository Structure (Current)

This repository includes:

```text
OllamaCloud/
  app.py
  blog.html
  compare_models.py
  model_comparison_report.md
  README.md
  requirements.txt
```

File goals:
- `app.py`: Streamlit chat UI to interactively test Ollama Cloud models.
- `compare_models.py`: quick multi-model comparison script (latency + basic output heuristics) and it writes `model_comparison_report.md`.
- `requirements.txt`: Python dependencies for both samples.
- `blog.html`: long-form Turkish blog post version of the tutorial for Blogger.
- `model_comparison_report.md`: an example output report generated by `compare_models.py` (you can regenerate it any time).

Dependency note:
- `compare_models.py` and `app.py` are **independent sample programs**. They do not import each other.
- They share only the same Python dependencies (`requirements.txt`) and the same authentication approach (an `OLLAMA_API_KEY` passed to the Ollama Cloud client).

What you achieved in this step: You understand what each file in the repo is for and which files depend on each other.

## 9. Clone and Run Locally

Clone the repository:

```powershell
git clone https://github.com/kmkarakaya/OllamaCloud.git
cd OllamaCloud
```

Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Set your API key (current shell session):

```powershell
$env:OLLAMA_API_KEY = "your_api_key_here"
```

Run the samples:

```powershell
python compare_models.py
streamlit run app.py
```

What you achieved in this step: You can clone the repo and run both Python samples end-to-end.

## 10. Contributing

Contributions are welcome (docs improvements, fixes, and enhancements).

Suggested workflow:
1. Fork the repo on GitHub.
2. Create a feature branch.
3. Make your changes (keep steps copy-paste friendly for Windows PowerShell).
4. Open a Pull Request describing the change and why it helps learners.

Important guidelines:
- Do not commit secrets: never put `OLLAMA_API_KEY` (or any API key) into code, markdown, or commits.
- Keep official references accurate and link to the primary docs (for example: [Cloud](https://docs.ollama.com/cloud), [API](https://docs.ollama.com/api), [Authentication](https://docs.ollama.com/api/authentication)).
- If you change model IDs in examples, prefer explaining how to list the latest models via `/api/tags` instead of assuming static availability.

What you achieved in this step: You know how to contribute safely and in a way that keeps the tutorial maintainable.

## 11. References

- https://docs.ollama.com/cloud
- https://docs.ollama.com/cloud#python
- https://docs.ollama.com/windows
- https://docs.ollama.com/api
- https://docs.ollama.com/api/authentication
- https://docs.ollama.com/api/tags
- https://docs.ollama.com/api/chat
- https://github.com/ollama/ollama-python
