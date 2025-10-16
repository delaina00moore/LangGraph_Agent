# Travel Advisor Agent

Quick example of a LangGraph-based travel advisor that uses Google Gemini via `langchain-google-genai`.

## Quick start

1. Create and activate a virtual environment
- macOS / Linux:
  ```sh
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Windows (PowerShell):
  ```ps
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

3. Set your Google AI API key:
- macOS / Linux:
  ```sh
  export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
  ```
- Windows (PowerShell):
  ```ps
  $env:GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
  ```
(The script will prompt if the variable is not set.)

4. Run the agent:
```sh
python travel_advisor.py
```
Type messages interactively; enter `exit` to quit.

## Notes
- The script binds tools to the model so the model can emit structured tool calls and uses a small state graph to orchestrate calls.
- The LLM fallback behaviour: if `GOOGLE_API_KEY` is missing, the script prompts for the key at startup.
