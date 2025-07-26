# Edison

Edison is a powerful Deep Research Intelligence package for Python that provides AI-powered research capabilities through both programmatic APIs and command-line interfaces.

## 🚀 Features

- **Dual Interface**: Both programmatic Python API and command-line interface.
- **Multiple Report Types**: Basic (2-5 pages) and Detailed (10-15 pages) research reports.
- **Multiple Output Formats**: JSON, Markdown, and Text.
- **Context Integration**: Include additional context from files.
- **Flexible Configuration**: Customizable models, temperature, and output options.
- **Professional Output**: Well-structured, comprehensive research reports.

## 📦 Installation

```bash
# Install Edison
pip install edison

# Or install in development mode with dev dependencies
pip install -e .[dev]
```

## 🛠 Quick Start

### Command Line Interface

```bash
# Set your OpenAI API key (option 1: environment variable)
export OPENAI_API_KEY="your-api-key-here"

# Or use an environment file (option 2: .env file)
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Generate a basic research report
edison --prompt "Research the latest trends in artificial intelligence" --basic

# Generate a detailed report with custom settings
edison --prompt "Deep analysis of renewable energy technologies" \
       --detailed \
       --model gpt-4 \
       --temperature 0.5 \
       --format markdown \
       --output-file energy_report.md

# Include additional context and use environment file
edison --prompt "Market analysis" \
       --context background_data.txt \
       --env .env \
       --detailed \
       --format json
```

### Python API

```python
import os
from edison import Edison

# Initialize Edison
edison = Edison(api_key=os.getenv("OPENAI_API_KEY"))

# Generate a research report
report = edison.generate_research_report(
    prompt="Research quantum computing trends",
    mode="basic",
    temperature=0.7
)

print(report)
```

## 📋 CLI Reference

### Commands

```bash
# Both interfaces work identically
edison --prompt "Your research question"
python -m edison --prompt "Your research question"
```

### Options

| Option          | Description                            | Default           | Required |
| --------------- | -------------------------------------- | ----------------- | -------- |
| `--prompt`      | Research prompt or question            | -                 | ✅       |
| `--basic`       | Generate basic report (2-5 pages)      | ✅ (default)      | ❌       |
| `--detailed`    | Generate detailed report (10-15 pages) | ❌                | ❌       |
| `--model`       | LLM model to use                       | `gpt-4`           | ❌       |
| `--temperature` | Creativity setting (0.0-2.0)           | `0.7`             | ❌       |
| `--format`      | Output format (json/markdown/text)     | `markdown`        | ❌       |
| `--output-file` | Save to file instead of stdout         | -                 | ❌       |
| `--context`     | Additional context file (.txt/.md)     | -                 | ❌       |
| `--api-key`     | OpenAI API key                         | `$OPENAI_API_KEY` | ✅       |

### Examples

```bash
# Basic research report
edison --prompt "AI trends in healthcare" --basic

# Detailed analysis with custom model
edison --prompt "Climate change impact analysis" \
       --detailed \
       --model gpt-4 \
       --temperature 0.3

# JSON output to file
edison --prompt "Market research for EVs" \
       --basic \
       --format json \
       --output-file market_analysis.json

# Include context from file with environment file
edison --prompt "Product strategy analysis" \
       --context company_data.md \
       --env .env \
       --detailed \
       --output-file strategy_report.md
```

## 📚 Documentation

- **[Usage Guide](docs/USAGE.md)** - Comprehensive usage instructions and examples.
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing and development setup.

## 📖 API Documentation

### Edison Class

```python
class Edison:
    def __init__(self, api_key: str, model: str = "gpt-4")

    def generate_text_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str

    def generate_research_report(
        self,
        prompt: str,
        mode: str = "basic",
        temperature: float = 0.7,
        context: Optional[str] = None
    ) -> str
```

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Run `make validate` to ensure quality.
5. Submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/thehackersplaybook/edison).

---

**Edison - Deep Research Intelligence for Python** 🚀
