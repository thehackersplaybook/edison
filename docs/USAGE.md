# Edison Usage Guide

Complete guide for using Edison, Deep Research Intelligence for Python.

## 🚀 Quick Start

### Installation

```bash
# Install Edison
pip install edison

# Or install in development mode
pip install -e .[dev]
```

### Set up your API key

Choose one of these methods:

**Option 1: Environment Variable**

```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option 2: Environment File (.env)**

```bash
# Create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Use with --env flag
edison --prompt "Your research question" --env .env
```

### Your First Report

```bash
# Generate a basic research report
edison --prompt "Research the latest trends in artificial intelligence" --basic

# Generate a detailed analysis
edison --prompt "Deep analysis of renewable energy technologies" --detailed --env .env
```

## 📋 CLI Reference

### Command Syntax

```bash
edison [OPTIONS] --prompt "Your research question"
# or
python -m edison [OPTIONS] --prompt "Your research question"
```

### Required Arguments

| Argument   | Description                 | Example                              |
| ---------- | --------------------------- | ------------------------------------ |
| `--prompt` | Research prompt or question | `--prompt "AI trends in healthcare"` |

### Optional Arguments

| Option          | Description                            | Default      | Example                   |
| --------------- | -------------------------------------- | ------------ | ------------------------- |
| `--basic`       | Generate basic report (2-5 pages)      | ✅ (default) | `--basic`                 |
| `--detailed`    | Generate detailed report (10-15 pages) | ❌           | `--detailed`              |
| `--model`       | LLM model to use                       | `gpt-4`      | `--model gpt-4`           |
| `--temperature` | Creativity setting (0.0-2.0)           | `0.7`        | `--temperature 0.5`       |
| `--format`      | Output format                          | `markdown`   | `--format json`           |
| `--output-file` | Save to file instead of stdout         | -            | `--output-file report.md` |
| `--context`     | Additional context file (.txt/.md)     | -            | `--context data.txt`      |
| `--env`         | Environment file path                  | -            | `--env .env`              |

## 📖 Detailed Usage Examples

### Basic Reports.

Generate quick, concise reports (2-5 pages):

```bash
# Simple basic report
edison --prompt "Current state of machine learning"

# Basic report with specific model
edison --prompt "Blockchain trends 2024" --model gpt-4 --temperature 0.3

# Basic report to file
edison --prompt "Cloud computing overview" --output-file cloud_report.md
```

### Detailed Reports.

Generate comprehensive analyses (10-15 pages):

```bash
# Detailed research report
edison --prompt "Impact of AI on healthcare" --detailed

# Detailed with custom settings
edison --prompt "Future of renewable energy" \
       --detailed \
       --model gpt-4 \
       --temperature 0.4 \
       --format markdown \
       --output-file energy_analysis.md
```

### Output Formats.

Choose from three output formats:

**Markdown (default)**

```bash
edison --prompt "Research topic" --format markdown
```

**JSON**

```bash
edison --prompt "Research topic" --format json --output-file report.json
```

**Plain Text**

```bash
edison --prompt "Research topic" --format text --output-file report.txt
```

### Using Context Files.

Provide additional context to enhance research quality:

```bash
# Prepare context file
echo "Company background information..." > company_context.txt

# Use context in research
edison --prompt "Market analysis for our company" \
       --context company_context.txt \
       --detailed \
       --output-file market_analysis.md
```

**Supported context formats:**

- `.txt` - Plain text files
- `.md` - Markdown files

### Environment Configuration.

#### Using .env Files

Create a `.env` file for your project:

```bash
# .env file contents
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_ORG_ID=your-org-id-here  # Optional
EDISON_DEFAULT_MODEL=gpt-4      # Optional
EDISON_DEFAULT_TEMPERATURE=0.7  # Optional
```

Use the environment file:

```bash
edison --prompt "Research question" --env .env
```

#### Global Environment

Set environment variables globally:

```bash
# In your shell profile (.bashrc, .zshrc, etc.)
export OPENAI_API_KEY="your-api-key"
export EDISON_DEFAULT_MODEL="gpt-4"
```

## 🎯 Use Cases & Examples.

### Academic Research.

```bash
# Literature review
edison --prompt "Recent advances in quantum computing algorithms" \
       --detailed \
       --temperature 0.3 \
       --output-file quantum_review.md

# Research proposal background
edison --prompt "Current challenges in natural language processing" \
       --context previous_research.md \
       --detailed \
       --format markdown
```

### Business Intelligence

```bash
# Market analysis
edison --prompt "Electric vehicle market trends and forecasts" \
       --detailed \
       --model gpt-4 \
       --output-file ev_market_report.md

# Competitive analysis
edison --prompt "AI startup landscape analysis" \
       --context company_data.txt \
       --detailed \
       --format json \
       --output-file competitive_analysis.json
```

### Technology Research.

```bash
# Technology overview
edison --prompt "Comparison of modern web frameworks" \
       --basic \
       --temperature 0.4

# Deep technical analysis
edison --prompt "Blockchain scalability solutions and trade-offs" \
       --detailed \
       --temperature 0.2 \
       --output-file blockchain_scalability.md
```

### Content Creation.

```bash
# Blog post research
edison --prompt "Benefits and challenges of remote work" \
       --basic \
       --format markdown \
       --output-file remote_work_research.md

# White paper background
edison --prompt "Enterprise AI adoption patterns and best practices" \
       --detailed \
       --context industry_data.txt \
       --output-file ai_adoption_whitepaper.md
```

## ⚙️ Advanced Configuration.

### Temperature Settings.

Control creativity vs. accuracy:

- **0.0-0.3**: Highly focused, factual, deterministic.
- **0.4-0.7**: Balanced creativity and accuracy (default: 0.7).
- **0.8-1.0**: More creative, exploratory.
- **1.1-2.0**: Highly creative, experimental.

```bash
# Conservative/factual approach
edison --prompt "Financial analysis" --temperature 0.2

# Balanced approach (default)
edison --prompt "Innovation trends" --temperature 0.7

# Creative exploration
edison --prompt "Future scenarios" --temperature 1.2
```

### Model Selection.

Choose the appropriate model for your needs:

```bash
# Standard model (default)
edison --prompt "Research topic" --model gpt-4

# Latest model
edison --prompt "Research topic" --model gpt-4-turbo

# Cost-effective option
edison --prompt "Research topic" --model gpt-3.5-turbo
```

## 🔧 Troubleshooting.

### Common Issues.

**API Key Not Found**

```bash
❌ Error: OpenAI API key required.
   Set OPENAI_API_KEY environment variable or use --env option
```

**Solution:** Set your API key using environment variable or .env file.

**Environment File Not Found.**

```bash
❌ Error: Environment file not found: .env
```

**Solution:** Create the .env file or check the path

**Invalid Temperature.**

```bash
❌ Error: Temperature must be between 0.0 and 2.0
```

**Solution:** Use a temperature value between 0.0 and 2.0

**Invalid Format.**

```bash
edison: error: argument --format: invalid choice: 'xml' (choose from 'json', 'markdown', 'text')
```

**Solution:** Use one of the supported formats: json, markdown, text.

### Debug Mode.

Add verbose output for debugging:

```bash
# Check environment loading
edison --prompt "test" --env .env  # Shows: 📄 Loaded environment from: .env

# Verify model initialization
edison --prompt "test" --env .env  # Shows: 🤖 Initialized Edison with model: gpt-4
```

## 📁 File Organization.

### Recommended Project Structure.

```
your-project/
├── .env                    # API keys and configuration
├── context/               # Context files
│   ├── company_info.md
│   └── market_data.txt
├── reports/               # Generated reports
│   ├── ai_trends.md
│   └── market_analysis.json
└── scripts/               # Automation scripts
    └── generate_reports.sh
```

### Example Automation Script.

```bash
#!/bin/bash
# scripts/generate_reports.sh

# Generate multiple reports
edison --prompt "Q4 market trends" --env .env --detailed --output-file reports/q4_trends.md
edison --prompt "Competitive landscape" --context context/company_info.md --env .env --detailed --output-file reports/competition.md
edison --prompt "Technology roadmap" --env .env --basic --format json --output-file reports/tech_roadmap.json

echo "✅ All reports generated successfully!"
```

## 📚 Integration Examples.

### With Other Tools.

**Combine with file processing:**

```bash
# Extract context from PDF (using external tool)
pdftotext document.pdf context.txt

# Generate report with extracted context
edison --prompt "Analyze this document" --context context.txt --detailed
```

**Pipeline with data analysis:**

```bash
# Generate research report
edison --prompt "Market analysis" --format json --output-file analysis.json

# Process with jq or other JSON tools
cat analysis.json | jq '.content' > processed_analysis.txt
```

### In Scripts and Automation.

**Python integration:**

```python
import subprocess
import json

def generate_research_report(prompt, output_file):
    cmd = [
        "edison",
        "--prompt", prompt,
        "--env", ".env",
        "--detailed",
        "--format", "json",
        "--output-file", output_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        with open(output_file, 'r') as f:
            return json.load(f)
    else:
        raise Exception(f"Error: {result.stderr}")

# Use in your application
report = generate_research_report("AI trends", "ai_report.json")
```

## 🔗 Additional Resources

- **Development Guide:** See `docs/DEVELOPMENT.md` for contributing.
- **API Reference:** See main README.md for Python API usage.
- **Examples:** Check the `examples/` directory for more use cases.
- **Issues:** Report bugs at GitHub repository.

---

**Edison - Deep Research Intelligence for Python** 🚀
