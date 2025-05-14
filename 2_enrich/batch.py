from __future__ import annotations

import time
from typing import Literal, List
from pathlib import Path
import json
import logging
import concurrent.futures
import random
import threading
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel, Field


def _genai_client() -> genai.Client:
    """
    Lazily initialise the Gemini client once per instance.
    The API key must be supplied as the environment variable GEMINI_API_KEY.
    """
    api_key = ""      # ← secret stays outside code
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)

# --------------------------------------------------------------------------- #
# Pydantic schemas                                                             #
# --------------------------------------------------------------------------- #

class FeatureWithTimestamp(BaseModel):
    text: str = Field(..., description="Feature description text")
    start_time: float | None = Field(
        None, description="Start time of the feature in seconds"
    )
    end_time: float | None = Field(
        None, description="End time of the feature in seconds"
    )

class Classification(BaseModel):
    is_sludge: bool
    layout_category: Literal[
        "horizontal", "vertical", "picture-in-picture",
        "grid", "layered", "collage", "other", "N/A"
    ] | None
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., min_length=100)
    text_features: List[FeatureWithTimestamp]
    visual_features: List[FeatureWithTimestamp]
    recommendations: List[str]

class VideoClassification(BaseModel):
    video_id: str
    classification: Classification

# --------------------------------------------------------------------------- #
# Prompt (unchanged – truncated for brevity here, but keep full text in code) #
# --------------------------------------------------------------------------- #

CONTEXT = """
Sludge content refers to a trend where multiple unrelated videos are played
simultaneously in a split-screen format, often designed to hold viewers' attention
through constant stimulation and prevent scrolling away. This format has become
increasingly prevalent on platforms like TikTok, YouTube Shorts, and Instagram Reels.
We will also record timestamps (in seconds) for each observed feature.
"""

CHARACTERISTICS = """
Sludge content typically displays these characteristics, all of which should be tagged
with start_time and end_time:

    SPLIT_SCREEN: Two or more videos displayed side by side or in various layouts
    MULTIPLE_UNRELATED_CLIPS: Each clip is often distinct and unrelated to others
    VARIED_FORMS: May include ASMR, TV clips, gameplay footage, collages, overlays
    ATTENTION RETENTION: Designed to maintain viewer attention through constant stimulation
    VISUAL DENSITY: High information density with multiple visual elements competing for attention
    ADDICTIVE DESIGN: Utilizes techniques to maximize engagement and viewing time


Non-sludge content may include:

    Single-frame videos with a cohesive focus
    Split-screen comparisons that are directly related
    Picture-in-picture for commentary or reaction purposes
    Educational comparisons with clear pedagogical intent


"""

TASK_INSTRUCTIONS = """
Analyze this video thoroughly and classify if it qualifies as sludge content.
For every feature you list—text or visual—provide its start_time and end_time in seconds.

For sludge content, the layout_category must be one of:
  - horizontal (side-by-side arrangement)
  - vertical (stacked arrangement)
  - picture-in-picture (smaller video(s) overlaid on larger one)
  - grid (multiple videos in a grid layout)
  - layered (videos overlapping in layers)
  - collage (mixed arrangement of videos)
  - other (for arrangements that don't fit the above)
For non-sludge content, set layout_category = "N/A".

Provide a detailed analysis including:

    Whether it's sludge content (is_sludge: true/false)
    The layout category as defined above
    Your confidence level (between 0 and 1)
    A comprehensive summary analyzing why this is or isn't sludge content
    Detailed visual features analysis with a timestamp range
    Detailed text features analysis with a timestamp range
    Recommendations for improvement or best practices


"""

IMPORTANT_NOTES = """
<IMPORTANT NOTES>

    Extract and process the audio and transcribe to text to analyze the textual features.
    Do not redefine what sludge means - use the provided definition.
    When referencing a specific visual or textual feature, include timestamps of when it appears
    Consider the totality of the content rather than focusing on single elements.
    Reason critically and guard against both false positives and false negatives.
    Provide detailed explanations for your classifications based on observable features.
    Be objective in your assessment without introducing bias against any particular content style.
    Consider the context and purpose of split screens (educational comparison vs. attention retention).
    Text analysis refers to captions, subtitles, or any visible text overlay.
    Don't just extract the text and write it but analyze it and reason if it's relevant to the content.
    Moreover, if the text is attention-grabbing, vulgar, educational, uneducational, distracting, inappropriate, or otherwise, it should be summarized in the text features.
    Don't include any personal, unimportant, or trackable information.
    Text and visual features should all be a maximum of 5 items, each summarized in 1 consistent and coherent sentence.
    Add recommendations based on the video content, don't hard copy the examples.
    Recommendations should be practical and actionable.
    Avoid using vague terms like "good" or "bad"; be specific about what works and what doesn't.
    Use clear and concise language, avoiding jargon or overly technical terms.
    DON'T PUT START_TIME AND END_TIME IF NOT NECESSARY.
    Do not invent fields.


</IMPORTANT NOTES>
"""

EXAMPLE_OUTPUT_SLUDGE = """
<GOOD SLUDGE CLASSIFICATION EXAMPLE>
{
  "is_sludge": true,
  "layout_category": "vertical",
  "confidence": 1.0,
  "summary": "The video is classified as sludge content due to its vertical split-screen layout presenting two entirely unrelated video streams simultaneously: a scene from the animated show South Park and gameplay from a mobile runner/simulator game. This format, combining disparate visual and auditory elements, is characteristic of content designed for attention retention through constant, unrelated stimulation.",
  "text_features": [
    {
      "text": "Captions of the South Park dialogue overlaid on the screen separator",
      "start_time": 0,
      "end_time": 60.0
    },
    {
      "text": "On-screen text and number overlays within the game footage indicating currency amounts, stat changes, and multipliers",
      "start_time": 0,
      "end_time": 60.0
    },
    {
      "text": "Top screen displays a scene from the animated show South Park",
      "start_time": 2.5,
      "end_time": 58.0
    }
    {
      "text": "Dialogue includes vulgar language",
      "start_time": 10.0,
      "end_time": 18.0
    },
  ],
  "recommendations": [
    "Avoid using unrelated video streams in a split-screen format.",
    "Using south park clips may subject the content to copyright claims.",
    "Consider using a single video stream with a cohesive theme.",
    "If using multiple clips, ensure they are related and serve a common purpose.",
    "Use overlays that enhance understanding rather than distract from the main content."
  ],
}
</GOOD SLUDGE CLASSIFICATION EXAMPLE>
"""

EXAMPLE_OUTPUT_NON_SLUDGE = """
<GOOD NON-SLUDGE CLASSIFICATION EXAMPLE>
{
  "is_sludge": false,
  "layout_category": "N/A",
  "confidence": 1.0,
  "summary": "The video is not sludge content. It features a single speaker delivering a monologue about technology addiction directly to the camera. There are no split screens, multiple unrelated video streams, or other characteristics defined as sludge content; the presentation is straightforward and focused on a single topic.",
  "text_features": [
    {
      "text": "A title overlay reads 'Why we get addicted to technology'.",
      "start_time": 0.0,
      "end_time": 4.2
    },
    {
      "text": "Subtitles transcribe key phrases spoken by the presenter (e.g., 'The reason', 'addicted', 'technology', 'make us feel good').",
      "start_time": 5.0,
      "end_time": 10.0
    },
    {
      "text": "The text reinforces the speaker's message and aids comprehension.",
      "start_time": 0,
      "end_time": 60.0
    },
    {
      "text": "The text is directly related to the spoken content.",
      "start_time": 0,
      "end_time": 60.0
    }
  ],
  "visual_features": [
    {
      "text": "The video primarily shows a single person speaking directly into a microphone.",
      "start_time": 0.0,
      "end_time": 60.0
    },
    {
      "text": "The visual focus remains consistently on the speaker and their message.",
      "start_time": 0.0,
      "end_time": 60.0
    },
    {
      "text": "No split-screen or multi-clip layouts are used.",
      "start_time": 0.0,
      "end_time": 60.0
    }
  ],
  "recommendations": [
    "No changes needed; the content is well-structured and focused."
  ]
}
</GOOD NON-SLUDGE CLASSIFICATION EXAMPLE>
"""

PROMPT = "\n".join(
    [
        CONTEXT,
        CHARACTERISTICS,
        TASK_INSTRUCTIONS,
        IMPORTANT_NOTES,
        EXAMPLE_OUTPUT_SLUDGE,
        EXAMPLE_OUTPUT_NON_SLUDGE,
    ]
)

# --------------------------------------------------------------------------- #
# Output paths & synchronisation                                              #
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "video_classifications.jsonl"

# lock to guard concurrent writes to OUTPUT_FILE
_FILE_WRITE_LOCK = threading.Lock()

def _load_processed_ids() -> set[str]:
    """Read OUTPUT_FILE (if any) and return set of already processed video_ids."""
    processed: set[str] = set()
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed.add(record.get("video_id"))
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue
    return processed

def classify_sludge_content(video_bytes: bytes):
    try:
        client = _genai_client()

        gen_resp = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=[
                types.Part(                       # video as inline blob
                    inline_data=types.Blob(
                        data=video_bytes,
                        mime_type="video/mp4"
                    )
                ),
                types.Part(text=PROMPT)
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": Classification,
                "temperature": 0.25,
                "max_output_tokens": 2048,
            },
        )

        return gen_resp.parsed

    except ClientError as err:
        # Gemini-specific errors
        return {"error": f"Gemini API error: {err}"}

    except Exception as err:                      # pylint: disable=broad-except
        return {"error": f"Unexpected error: {err}"}

def process_video(video_path: Path, semaphore: threading.Semaphore) -> None:
    """Process a single video file and append classification output to JSONL."""
    video_id = video_path.stem
    if video_id in _PROCESSED_IDS:
        logging.info("[SKIP] %s already present in output", video_path.name)
        return

    # read bytes once to avoid holding the file open too long
    try:
        video_bytes = video_path.read_bytes()
    except Exception as err:  # pylint: disable=broad-except
        logging.exception("[ERROR] Failed reading %s: %s", video_path, err)
        return

    max_attempts = 5
    delay = 0.25  # seconds
    for attempt in range(1, max_attempts + 1):
        try:
            with semaphore:  # limit concurrent requests
                result = classify_sludge_content(video_bytes)
            # if the API itself returns an error we treat it as failure
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(result["error"])

            # Pydantic BaseModel → dict
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            else:
                result_dict = result  # pragma: no cover

            # ensure video_id stored
            result_dict = {
                "video_id": video_id,
                **result_dict,
            }

            # atomic append to shared JSONL
            with _FILE_WRITE_LOCK:
                with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                    json.dump(result_dict, f, ensure_ascii=False)
                    f.write("\n")

            logging.info("[SUCCESS] %s", video_path.name)
            return

        except Exception as err:  # pylint: disable=broad-except
            logging.warning(
                "[RETRY %s/%s] %s failed: %s", attempt, max_attempts, video_path.name, err
            )
            if attempt == max_attempts:
                logging.error("[FAILED] %s: %s", video_path.name, err)
            else:
                # exponential backoff with jitter
                sleep_for = delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
                time.sleep(sleep_for)


def main() -> None:
    dataset_root = Path("/mnt/disks/hyper_ml_storage/skibidi-detector-workflow/sludge_dataset")
    categories = ["SLUDGE", "NON SLUDGE"]

    # Setup logging – separate success and error logs as requested
    logs_dir = Path(__file__).with_suffix("").parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "success.log", mode="a", encoding="utf-8"),
            logging.FileHandler(logs_dir / "errors.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    video_files: list[Path] = []
    global _PROCESSED_IDS  # noqa: PLW0603
    _PROCESSED_IDS = _load_processed_ids()
    for category in categories:
        category_dir = dataset_root / category
        if not category_dir.exists():
            logging.warning("Category directory missing: %s", category_dir)
            continue
        video_files.extend([
            vp for vp in category_dir.glob("*.mp4") if vp.stem not in _PROCESSED_IDS
        ])

    # Shuffle list to spread load more evenly across categories
    random.shuffle(video_files)

    # Semaphore to limit inflight Gemini requests to 6
    semaphore = threading.Semaphore(value=6)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_video, vp, semaphore) for vp in video_files]
        # Wait for completion – gather to propagate exceptions
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()
