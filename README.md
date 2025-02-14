# Edison: Deep Research

> 🚨 This project is educational, experimental and under active development. Use with caution.

Simple, effective &amp; powerful Deep Research capabilities in Python.

---

## Overview

Edison is a Python package with enables application developers integrate deep research capabilities within 10 lines of code. Edison is state-of-the-art, and uses the R3 Pattern (Reflect, Revise, Regenerate) to provide high-quality reports for given natural language queries.

---

## Usage

- [Install Python 3 and above.](https://www.python.org/downloads/)

- Install the edison package.

```bash
pip install edison
```

- Add a `.env` file with FireCrawl and OpenAI keys in the root folder of your script.

```text
OPENAI_API_KEY=your-openai-api-key
FIRECRAWL_API_KEY=your-firecrawl-api-key
```

- Write your first deep research program.

```python
# app.py
from edison import EdisonDeepResearch

researcher = EdisonDeepResearch()

topic = "The advancements in Machine Learning between 2010 to 2025."

report = researcher.deep(topic=topic, depth=0.5)

print(f"Report on {topic}")
print(report)
```

- Run the program.

```bash
python app.py
```

---

## Motivations & Objectives

- Build a simple and cost-effective deep research solution that's easily integrable.
- Contribute to a deep research package in Python for which there aren't many alternatives.
- Support the broader vision of building high-quality knowledge systems at The Hackers Playbook.
- Contribute to the Python and Open Source Communities with a novel product.
- Create teaching material for The Hackers Playbook System Design and Programming courses.

## Contributions

We welcome contributions from developers around the globe. The steps to contribute are simple:

- Fork the repository.
- Create a new branch with your changes.
- Submit a PR to this repository.
- Complete the PR review process with our team.

---

## License

Edison is distributed under the MIT License. Refer to the [LICENSE](https://github.com/thehackersplaybook/edison/blob/main/LICENSE) file for full details.
