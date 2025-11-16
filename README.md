# Narration Image Generator

A production-grade CLI tool that automatically generates high-resolution images
for narration scripts using the Pexels API and OpenAI for keyword generation.

## Features

- ✨ Automatically splits narration scripts into scenes (3-10 sentences each)
- 🔑 Generates 3-6 relevant keywords per scene using OpenAI
- 📸 Downloads 3-5 high-resolution images per scene from Pexels
- 📁 Creates organized directory structure for easy CapCut import
- 🛡️ Comprehensive error handling and logging
- 🚀 Modern Python stack with best practices

## Installation

### Prerequisites

- Python 3.8 or higher
- macOS (tested on macOS 14+)
- uv (install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Setup

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd images-faceless
   ```

2. **Install dependencies with uv**
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

3. **Download NLTK data** (required for text processing)
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

4. **Configure API keys**
   - Copy `.env.example` to `.env`
   - Add your Pexels API key
   - Add your OpenAI API key (optional, for better keyword generation)

## Usage

### Basic Usage

```bash
python narration_generator.py path/to/your/script.txt
```

### Example

```bash
python narration_generator.py examples/discipline_perception.txt
```

### Output Structure

```
output/
└── discipline_perception/
    ├── scene_01/
    │   ├── keywords.txt
    │   ├── image_01.jpg
    │   ├── image_02.jpg
    │   └── ...
    ├── scene_02/
    │   ├── keywords.txt
    │   ├── image_01.jpg
    │   └── ...
    └── ...
```

## Example Input

Create a file `script.txt`:

```
There's a remarkable truth hidden in the story of how some people thrive while others collapse under the same weight. We've all witnessed it. Two individuals face identical circumstances—one crumbles, paralyzed by fear, while the other moves forward with clarity and purpose.

Let me take you back to 1857, to a young bookkeeper in Cleveland named John D. Rockefeller. He's sixteen years old, the son of an alcoholic con artist who abandoned his family. He's working for fifty cents a day, celebrating what he calls "Job Day"—grateful just to have honest work.
```

## API Keys Required

- **Pexels API**: Get free API key at https://www.pexels.com/api/
- **OpenAI API**: Get API key at https://platform.openai.com/api-keys (optional)

## Configuration

Edit `.env` to customize:

```bash
DEFAULT_IMAGES_PER_SCENE=4  # 3-5 images per scene
MIN_SCENE_SENTENCES=3       # Minimum sentences per scene
MAX_SCENE_SENTENCES=10      # Maximum sentences per scene
```

## Troubleshooting

### Common Issues

1. **Permission denied**: Ensure script has execute permissions
   ```bash
   chmod +x narration_generator.py
   ```

2. **NLTK data missing**: Download required data
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

3. **API rate limits**: Wait a moment between large requests

## License

MIT License - feel free to use for personal and commercial projects.
