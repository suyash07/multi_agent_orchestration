# Multi-Agent System Starter

A short starter guide for setting up GitHub Copilot and a Python-based multi-agent workflow on macOS.

## 1. Install Copilot
- Install the VS Code extensions: GitHub Copilot and GitHub Copilot Chat.
- Sign in with your GitHub account.
- If you use the Copilot CLI, run `copilot` and then enter `/login`.

## 2. Set up Python environment
```bash
cd /Users/suyashmhetre/Desktop/Projects/multi_agent_system
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install openai langchain crewai autogen
```

## 3. Start building agents
- Create a simple `main.py`.
- Use separate roles such as planner, executor, and reviewer.
- Keep prompts clear and structured.

## 4. Run your app
```bash
python main.py
```

## Useful commands
```bash
source .venv/bin/activate
pip install openai langchain crewai autogen
```

If you want, the next step is to add a basic agent workflow file and a sample prompt flow.
