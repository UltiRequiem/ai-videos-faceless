#!/usr/bin/env python3

"""
Narration Image Generator CLI Tool

A production-grade CLI tool that automatically generates high-resolution images
for narration scripts using the Pexels API and OpenAI for keyword generation.

Usage:
    python narration_generator.py script.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
import nltk
from nltk.tokenize import sent_tokenize
from PIL import Image
from pathvalidate import sanitize_filename
from rich.console import Console
from rich.progress import Progress
from rich.logging import RichHandler

_ = load_dotenv()

console = Console()

PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEFAULT_IMAGES_PER_SCENE = int(os.getenv('DEFAULT_IMAGES_PER_SCENE', '4'))
MIN_SCENE_SENTENCES = int(os.getenv('MIN_SCENE_SENTENCES', '3'))
MAX_SCENE_SENTENCES = int(os.getenv('MAX_SCENE_SENTENCES', '10'))

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
PEXELS_HEADERS = {"Authorization": PEXELS_API_KEY}

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

class NarrationImageGenerator:
    """Main class for generating images from narration scripts."""

    script_path: Path
    script_name: str
    output_dir: Path
    use_nested_dirs: bool
    logger: logging.Logger

    def __init__(self, script_path: str, use_nested_dirs: bool = False):
        """Initialize the generator with a script path."""
        self.script_path = Path(script_path)
        self.script_name = sanitize_filename(self.script_path.stem)
        self.output_dir = Path("output") / self.script_name
        self.use_nested_dirs = use_nested_dirs

        self.setup_logging()

        self.validate_setup()

    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                RichHandler(rich_tracebacks=True),
                logging.FileHandler(log_dir / f"{self.script_name}.log")
            ]
        )

        self.logger = logging.getLogger(__name__)

    def validate_setup(self) -> None:
        """Validate that all required components are available."""
        if not self.script_path.exists():
            console.print(f"❌ Error: Script file not found: {self.script_path}", style="red")
            sys.exit(1)

        if not PEXELS_API_KEY:
            console.print("❌ Error: PEXELS_API_KEY not found in .env file", style="red")
            sys.exit(1)

        # Check if NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            console.print("📦 Downloading NLTK data...", style="yellow")
            nltk.download('punkt_tab', quiet=True)

        self.logger.info(f"Initialized generator for script: {self.script_path}")

    def load_script(self) -> str:
        """Load and clean the narration script."""
        try:
            with open(self.script_path, 'r', encoding='utf-8') as f:
                content = f.read()

            content = re.sub(r'\[Scene[^\]]*\]', '', content)  # Remove scene directions
            content = re.sub(r'#{1,6}\s+', '', content)  # Remove markdown headers
            content = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', content)  # Remove markdown emphasis
            content = re.sub(r'---+', '', content)  # Remove horizontal rules
            content = re.sub(r'\n{3,}', '\n\n', content)  # Normalize line breaks
            content = content.strip()

            self.logger.info(f"Loaded script with {len(content)} characters")

            return content

        except Exception as e:
            console.print(f"❌ Error loading script: {e}", style="red")
            sys.exit(1)

    def split_into_scenes(self, content: str) -> list[dict[str, any]]:
        """Split the script content into scenes with 3-10 sentences each."""

        sentences = sent_tokenize(content)
        scenes = []
        current_scene = []
        scene_number = 1

        for sentence in sentences:
            # Skip very short sentences (likely fragments)
            if len(sentence.strip()) < 10:
                continue

            current_scene.append(sentence.strip())

            # Create a scene when we have enough sentences
            if len(current_scene) >= MIN_SCENE_SENTENCES:
                # End scene if we hit max sentences or find a natural break
                if (len(current_scene) >= MAX_SCENE_SENTENCES or
                    self._is_scene_boundary(sentence)):

                    scene_text = ' '.join(current_scene)
                    scenes.append({
                        'number': scene_number,
                        'text': scene_text,
                        'sentences': len(current_scene),
                        'folder': f"scene_{scene_number:02d}"
                    })

                    current_scene = []
                    scene_number += 1

        # Add remaining sentences as final scene
        if current_scene:
            scene_text = ' '.join(current_scene)
            scenes.append({
                'number': scene_number,
                'text': scene_text,
                'sentences': len(current_scene),
                'folder': f"scene_{scene_number:02d}"
            })

        self.logger.info(f"Split script into {len(scenes)} scenes")
        return scenes

    def _is_scene_boundary(self, sentence: str) -> bool:
        """Determine if a sentence marks a natural scene boundary."""

        boundary_indicators = [
            'fast forward', 'years later', 'meanwhile', 'now here\'s',
            'let me take you', 'now imagine', 'picture this',
            'here\'s what', 'this is where', 'but then'
        ]

        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in boundary_indicators)

    def generate_keywords(self, scene_text: str) -> list[str]:
        """Generate 3-6 relevant keywords for a scene."""

        if openai_client:
            return self._generate_keywords_openai(scene_text)
        else:
            return self._generate_keywords_simple(scene_text)

    def _generate_keywords_openai(self, scene_text: str) -> list[str]:
        if not openai_client:
            raise ValueError("OpenAI client is not initialized.")

        """Generate keywords using OpenAI API."""
        try:
            prompt = f"""
            Extract 3-6 visual keywords from this narration text that would be good for finding stock photos.
            Focus on concrete, visual concepts that can be photographed.
            Avoid abstract concepts, emotions, or complex scenes.

            Text: {scene_text[:500]}...

            Return only the keywords, separated by commas:
            """

            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )

            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]

            keywords = keywords[:6] if len(keywords) > 6 else keywords
            if len(keywords) < 3:
                keywords.extend(self._generate_keywords_simple(scene_text)[:3-len(keywords)])

            return keywords

        except Exception as e:
            self.logger.warning(f"OpenAI keyword generation failed: {e}")
            return self._generate_keywords_simple(scene_text)

    def _generate_keywords_simple(self, scene_text: str) -> list[str]:
        """Generate keywords using simple text analysis."""

        # Common visual nouns that make good search terms
        visual_words = [
            'office', 'business', 'person', 'man', 'woman', 'building', 'city',
            'street', 'book', 'paper', 'desk', 'computer', 'money', 'hand',
            'face', 'eye', 'road', 'car', 'house', 'room', 'window', 'door',
            'tree', 'sky', 'water', 'mountain', 'field', 'crowd', 'people',
            'success', 'growth', 'leader', 'team', 'meeting', 'presentation'
        ]

        words = re.findall(r'\b[a-z]+\b', scene_text.lower())
        found_keywords = [word for word in visual_words if word in words]

        # If not enough visual words found, add some generic business/success terms
        if len(found_keywords) < 3:
            generic_terms = ['business success', 'professional growth', 'leadership', 'strategy']
            found_keywords.extend(generic_terms[:4-len(found_keywords)])

        return found_keywords[:6]

    def download_images_for_scene(self, scene: dict, keywords: list[str]) -> list[str]:
        """Download 3-5 high-resolution images for a scene."""
        if self.use_nested_dirs:
            # Original nested structure
            scene_dir = self.output_dir / scene['folder']
            scene_dir.mkdir(parents=True, exist_ok=True)
            image_prefix = ""
        else:
            # Flat structure - all images in root output directory
            scene_dir = self.output_dir
            scene_dir.mkdir(parents=True, exist_ok=True)
            image_prefix = f"scene_{scene['number']:02d}_"

        downloaded_images = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit download tasks for each keyword
            future_to_keyword = {}
            for i, keyword in enumerate(keywords[:DEFAULT_IMAGES_PER_SCENE]):
                future = executor.submit(self._search_and_download_image,
                                       keyword, scene_dir, i+1, scene['number'], image_prefix)
                future_to_keyword[future] = keyword

            # Collect results
            for future in as_completed(future_to_keyword):
                keyword = future_to_keyword[future]
                try:
                    image_path = future.result()
                    if image_path:
                        downloaded_images.append(image_path)
                        console.print(f"  ✅ Downloaded image for '{keyword}'", style="green")
                    else:
                        console.print(f"  ⚠️  No suitable image found for '{keyword}'", style="yellow")
                except Exception as e:
                    self.logger.error(f"Error downloading image for '{keyword}': {e}")

        return downloaded_images

    def _search_and_download_image(self, keyword: str, scene_dir: Path, image_number: int, scene_number: int = 0, prefix: str = "") -> str | None:
        """Search for and download a single image."""
        try:
            # Search for images
            params = {
                'query': keyword,
                'per_page': 5,
                'page': 1,
                'size': 'large',
                'orientation': 'landscape'
            }

            response = requests.get(PEXELS_SEARCH_URL, headers=PEXELS_HEADERS, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if not data.get('photos'):
                return None

            # Get the first suitable image
            photo = data['photos'][0]
            image_url = photo['src']['large']

            # Download the image
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()

            # Save the image with appropriate filename
            if prefix:
                # Flat structure: scene_01_image_01.jpg
                image_filename = f"{prefix}image_{image_number:02d}.jpg"
            else:
                # Nested structure: image_01.jpg
                image_filename = f"image_{image_number:02d}.jpg"

            image_path = scene_dir / image_filename

            with open(image_path, 'wb') as f:
                f.write(img_response.content)

            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception:
                image_path.unlink()  
                return None

            return str(image_path)

        except Exception as e:
            self.logger.error(f"Error in _search_and_download_image for '{keyword}': {e}")
            return None

    def save_keywords(self, scene_dir: Path, keywords: list[str], scene_number: int = 0) -> None:
        """Save keywords to a text file in the scene directory."""
        if self.use_nested_dirs:
            keywords_file = scene_dir / "keywords.txt"
        else:
            keywords_file = scene_dir / f"scene_{scene_number:02d}_keywords.txt"

        with open(keywords_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(keywords))

    def generate_all_keywords(self, scenes: list[dict[str, any]]) -> list[dict[str, any]]:
        """Generate keywords for all scenes in batch."""
        console.print("\n🔑 Generating keywords for all scenes...", style="bold blue")

        with Progress() as progress:
            task = progress.add_task("Generating keywords...", total=len(scenes))

            for scene in scenes:
                console.print(f"📋 Scene {scene['number']}: {scene['sentences']} sentences", style="cyan")

                # Generate keywords
                keywords = self.generate_keywords(scene['text'])
                scene['keywords'] = keywords

                console.print(f"  🔑 Keywords: {', '.join(keywords)}", style="blue")

                # Save keywords to file
                if self.use_nested_dirs:
                    scene_dir = self.output_dir / scene['folder']
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    self.save_keywords(scene_dir, keywords, scene['number'])
                else:
                    self.save_keywords(self.output_dir, keywords, scene['number'])

                progress.update(task, advance=1)

                # Small delay for OpenAI API rate limiting
                time.sleep(0.2)

        console.print("✅ All keywords generated!", style="bold green")
        return scenes

    def download_all_images(self, scenes: list[dict[str, any]]) -> None:
        """Download images for all scenes in batch."""
        console.print("\n📸 Downloading images for all scenes...", style="bold blue")

        # Count total images to download
        total_images = len(scenes) * DEFAULT_IMAGES_PER_SCENE
        console.print(f"📊 Planning to download ~{total_images} images total", style="yellow")

        with Progress() as progress:
            task = progress.add_task("Downloading images...", total=len(scenes))

            for scene in scenes:
                console.print(f"\n🎬 Scene {scene['number']}: Downloading {DEFAULT_IMAGES_PER_SCENE} images", style="cyan")

                # Download images for this scene
                downloaded_images = self.download_images_for_scene(scene, scene['keywords'])

                console.print(f"  ✅ Downloaded {len(downloaded_images)} images", style="green")
                progress.update(task, advance=1)

                # Rate limiting for Pexels API
                time.sleep(0.5)

        console.print("✅ All images downloaded!", style="bold green")

    def generate_all_images(self) -> None:
        """Main method to process the entire script and generate all images."""
        console.print(f"🎬 Processing script: {self.script_path}", style="bold blue")

        content = self.load_script()
        scenes = self.split_into_scenes(content)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"📊 Found {len(scenes)} scenes to process", style="yellow")

        scenes_with_keywords = self.generate_all_keywords(scenes)

        self.download_all_images(scenes_with_keywords)

        console.print(f"\n🎉 All done! Generated images for {len(scenes)} scenes", style="bold green")
        console.print(f"📁 Output directory: {self.output_dir.absolute()}", style="blue")

        if self.use_nested_dirs:
            console.print(f"📂 Structure: Nested folders (scene_01/, scene_02/, etc.)", style="cyan")
        else:
            console.print(f"📂 Structure: Flat directory (all images at same level)", style="cyan")

        console.print(f"💡 Ready to import into CapCut!", style="bold yellow")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate high-resolution images for narration scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python narration_generator.py script.txt
  python narration_generator.py examples/discipline_perception.txt

Output:
  Creates organized directory structure in output/<script_name>/
  Each scene contains keywords.txt and 3-5 high-resolution images
        """
    )

    parser.add_argument(
        'script_path',
        help='Path to the narration script (.txt file)'
    )

    parser.add_argument(
        '--nested',
        action='store_true',
        help='Use nested directory structure (scene_01/, scene_02/, etc.). Default is flat structure.'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Narration Image Generator 1.0.0'
    )

    args = parser.parse_args()

    try:
        generator = NarrationImageGenerator(args.script_path, use_nested_dirs=args.nested)
        generator.generate_all_images()

    except KeyboardInterrupt:
        console.print("\n❌ Operation cancelled by user", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
