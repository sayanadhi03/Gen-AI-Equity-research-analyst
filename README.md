# Gen AI Equity Research Analyst

A small prototype that demonstrates how to build a generative-AI powered equity research assistant. It contains a lightweight script, example notebooks, and sample data to explore retrieval and summarization workflows.

**Key files**
- `main.py`: example entry-point to run core functionality
- `requirements.txt`: Python dependencies
- `notebooks/`: Jupyter notebooks and sample data

**Quick overview**
- Purpose: Provide tools and examples for retrieving, analyzing, and summarizing equity-related text using retrieval + generative models.

## Prerequisites
- Windows (instructions below assume PowerShell)
- Python 3.8 or newer

## Setup (Windows)
1. Open PowerShell in the repository root.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If you use a different shell (Command Prompt, Git Bash, WSL), adapt the activation command accordingly.

## OpenAI API key (required for model access)

Some notebooks and scripts expect an OpenAI API key to call model APIs. Set your key using one of these approaches.

- Temporary (current PowerShell session):

```powershell
$env:OPENAI_API_KEY = "sk-REPLACE_WITH_YOUR_KEY"
```

- Persist for your Windows user (keeps key across sessions):

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-REPLACE_WITH_YOUR_KEY", "User")
```

- Use a `.env` file (recommended for local development, but do NOT commit this file):

1. Create a file named `.env` in the repository root with the line:

```
OPENAI_API_KEY=sk-REPLACE_WITH_YOUR_KEY
```

2. Load this in Python (example using `python-dotenv`):

```python
from dotenv import load_dotenv
load_dotenv()
import os
key = os.getenv("OPENAI_API_KEY")
```

Security notes:
- Never commit your API key to version control. Add `.env` to your `.gitignore`.
- Treat the key like a password and rotate it if it is exposed.

## Run the example script
- To run the main example:

```powershell
python main.py
```

- `main.py` is a simple runner for experiments — open it to see what inputs and outputs it expects. Modify or extend it for your use case.

## Working with the notebooks
- Install JupyterLab if you haven't already:

```powershell
pip install jupyterlab
jupyter lab
```

- Open the `notebooks/` folder in JupyterLab. Notebooks of interest:
	- `faiss.ipynb` — vector indexing and search examples
	- `retrival.ipynb` — retrieval pipeline experiments
	- `MyFirstNotebook.ipynb` — basic exploration and examples

## Data
- Example data files live in `notebooks/` and include `movies.csv`, `nvda_news_1.txt`, and `sample_text.csv`.
- Replace or extend these files with your own datasets for experiments.

## Troubleshooting
- If a package fails to install, ensure your venv is active and Python version >= 3.8.
- On permission errors when creating the venv, run PowerShell as Administrator or choose a different folder.
- If Jupyter won't start, try `python -m jupyter lab` to ensure it uses the active venv.

## Contributing
- Issues and PRs are welcome. Suggested contributions:
	- Additional notebooks demonstrating analysis workflows
	- Scripts to preprocess real-world financial text
	- Integration with a model API or larger retriever/backends

## Next steps (suggested)
- Run `python main.py` to see example output.
- Open `notebooks/retrival.ipynb` and run cells to explore retrieval + summarization.
- Add your own documents in `notebooks/` and adapt the retrieval code.

## Contact
- Maintainer: (update this line with your name and contact information)
