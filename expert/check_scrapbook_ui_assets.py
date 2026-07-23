from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from scrapbook_sidebar import EMOTIONS, ScrapbookSidebar

UI_ASSETS = BASE / "assets" / "ui"
OUTPUT = UI_ASSETS / "previews" / "sidebar_contact_sheet_runtime.png"


def main() -> None:
    renderer = ScrapbookSidebar(
        asset_root=UI_ASSETS,
        width=320,
        height=720,
        left_overlap=72,
    )

    confidence_by_emotion = {
        "angry": 0.81,
        "disgust": 0.69,
        "fear": 0.71,
        "happy": 0.86,
        "neutral": 0.63,
        "sad": 0.74,
        "surprise": 0.78,
    }

    cards = []
    for emotion in EMOTIONS:
        card_rgba = renderer.render(
            display_label=emotion,
            confidence=confidence_by_emotion[emotion],
            preview_label=emotion,
            effects_enabled=True,
            face_detected=True,
            detected_label=emotion,
            raw_label=emotion,
            prediction_ms=24.6,
            prediction_p95_ms=28.8,
            pose_ms=23.9,
            classifier_ms=0.31,
            display_fps=30.0,
            debug_enabled=False,
        )
        cards.append((emotion, Image.fromarray(card_rgba, mode="RGBA")))

    margin = 24
    label_height = 36
    columns = 4
    rows = 2
    card_w = renderer.width
    card_h = renderer.height
    sheet = Image.new(
        "RGB",
        (
            margin + columns * (card_w + margin),
            margin + rows * (card_h + label_height + margin),
        ),
        (238, 233, 224),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (emotion, card) in enumerate(cards):
        row = index // columns
        col = index % columns
        x = margin + col * (card_w + margin)
        y = margin + row * (card_h + label_height + margin)
        sheet.paste(card, (x, y), card)
        draw.text(
            (x + 8, y + card_h + 10),
            emotion.upper(),
            font=font,
            fill=(40, 40, 42),
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT)
    print(f"Scrapbook UI asset check passed: {OUTPUT}")


if __name__ == "__main__":
    main()
