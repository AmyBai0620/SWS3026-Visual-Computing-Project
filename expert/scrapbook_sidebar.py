from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


EMOTIONS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

# RGB colors used only for runtime-drawn accents.
EMOTION_ACCENTS = {
    "angry": (214, 77, 65),
    "disgust": (91, 143, 91),
    "fear": (116, 101, 157),
    "happy": (235, 177, 18),
    "neutral": (92, 91, 86),
    "sad": (75, 137, 202),
    "surprise": (253, 142, 42),
}

TEXT_DARK = (40, 39, 36, 255)
TEXT_MUTED = (78, 76, 70, 255)


class ScrapbookSidebar:
    """Render the scrapbook sidebar with a left-side camera overlap."""

    def __init__(
        self,
        asset_root: Path,
        width: int = 320,
        height: int = 720,
        left_overlap: int = 72,
    ) -> None:
        self.asset_root = Path(asset_root)

        # ``width`` is the original sidebar content width.  ``left_overlap``
        # adds extra paper only on the left, so the sidebar can overlap the
        # camera while its right edge remains fixed.
        self.content_width = int(width)
        self.left_overlap = max(0, int(left_overlap))
        self.width = self.content_width + self.left_overlap
        self.height = int(height)

        # The widened panel does not necessarily shift the original paper
        # content by the full overlap amount, because the source PNG has a
        # transparent left margin and is cropped again.  _prepare_base()
        # calculates the real shift and stores it here.
        self.content_offset_x = self.left_overlap

        # Small layout corrections based on the real camera preview.
        # The metric labels need a little breathing room below the first line.
        # The shortcut rows use tighter spacing so the last row stays above
        # the decorative bottom strip without crowding the paper tabs.
        # Move the runtime metrics slightly upward so the two rows sit
        # more evenly between the hand-drawn separator lines.
        self.metrics_y_shift = -4

        # The shortcut block was raised from the original layout, then
        # lowered a touch after checking the full camera preview.
        self.shortcut_heading_y = 568
        self.shortcut_start_y = 583
        self.shortcut_line_gap = 14

        # Shift the foreground slightly farther right.  The title and
        # confidence area move gently, while the middle metrics receive an
        # additional nudge so they no longer crowd the torn-paper edge.
        # The paper base and bottom emotion strip stay fixed.
        self.foreground_shift_x = 32
        self.metrics_extra_shift_x = 6

        # Fine-tune the two values printed around the paper tabs.
        # "Automatic" only moves a little from the previous layout, while
        # the short "On/Off" value follows the larger foreground shift.
        self.mode_value_shift_x = -8
        self.effects_value_shift_x = 6

        # Fine-tune only the words printed on the two paper tabs.  Both
        # move right slightly; EFFECTS needs the larger correction.
        self.mode_label_shift_x = 4
        self.effects_label_shift_x = 9
        self.tab_label_y_shift = -11

        # Anchor the decorative cloud to the real lower-right corner of the
        # widened paper, instead of the original 320 px content box.  A small
        # overlap with the coloured bottom strip makes it feel pasted on.
        self.cloud_right_inset = 6
        self.cloud_bottom_strip_overlap = 18

        # Keep the left margin as-is, but let the strip sit closer to the
        # panel's right edge.  Separate values make the visual spacing easier
        # to fine-tune without moving the left side.
        self.bottom_strip_left_inset = 8
        self.bottom_strip_right_inset = 2

        self.font_regular = self._load_font(15, bold=False)
        self.font_small = self._load_font(13, bold=False)
        self.font_tiny = self._load_font(12, bold=False)
        self.font_bold = self._load_font(18, bold=True)
        self.font_percent = self._load_font(44, bold=True)

        self.base = self._prepare_base(
            self._open_rgba(
                self.asset_root / "sidebar" / "base" / "sidebar_base.png"
            )
        )

        self.titles = {}
        self.bottom_strips = {}
        self.bottom_strip_positions = {}
        self.clouds = {}

        for emotion in EMOTIONS:
            title_image = self._open_rgba(
                self.asset_root / "sidebar" / "titles" / f"{emotion}.png"
            )
            title_max_width = 314 if emotion == "surprise" else 282
            title_max_height = 128 if emotion == "surprise" else 112
            if emotion == "surprise":
                # The source SURPRISE PNG contains generous transparent margins.
                # Trim them first so the visible lettering grows noticeably.
                self.titles[emotion] = self._trim_and_fit(
                    title_image,
                    max_width=title_max_width,
                    max_height=title_max_height,
                    padding=10,
                )
            else:
                self.titles[emotion] = self._fit_inside(
                    title_image,
                    max_width=title_max_width,
                    max_height=title_max_height,
                )

            strip = self._open_rgba(
                self.asset_root
                / "sidebar"
                / "bottom_strips"
                / f"{emotion}.png"
            )
            # First estimate the strip height, then align the strip to the
            # visible paper near the bottom with independent left and right insets.
            estimated_height = max(
                1, round(strip.height * self.content_width / strip.width)
            )
            paper_left, paper_right = self._bottom_paper_bounds(estimated_height)
            strip_x = min(
                self.width - 1, paper_left + self.bottom_strip_left_inset
            )
            strip_right = max(
                strip_x + 1, paper_right - self.bottom_strip_right_inset
            )
            strip_right = min(self.width, strip_right)
            strip_width = max(1, strip_right - strip_x)
            strip_height = max(
                1, round(strip.height * strip_width / strip.width)
            )

            # Recalculate once using the final strip height for better
            # alignment with the irregular torn-paper edge.
            paper_left, paper_right = self._bottom_paper_bounds(strip_height)
            strip_x = min(
                self.width - 1, paper_left + self.bottom_strip_left_inset
            )
            strip_right = max(
                strip_x + 1, paper_right - self.bottom_strip_right_inset
            )
            strip_right = min(self.width, strip_right)
            strip_width = max(1, strip_right - strip_x)
            strip_height = max(
                1, round(strip.height * strip_width / strip.width)
            )

            self.bottom_strip_positions[emotion] = strip_x
            self.bottom_strips[emotion] = strip.resize(
                (strip_width, strip_height),
                Image.Resampling.LANCZOS,
            )

            cloud = self._open_rgba(
                self.asset_root
                / "sidebar"
                / "clouds"
                / f"{emotion}_main_cloud.png"
            )
            self.clouds[emotion] = self._trim_and_fit(
                cloud,
                max_width=126,
                max_height=92,
                padding=5,
            )

    @staticmethod
    def _open_rgba(path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Missing scrapbook UI asset: {path}")
        return Image.open(path).convert("RGBA")

    @staticmethod
    def _font_candidates(bold: bool):
        if bold:
            names = (
                r"C:\Windows\Fonts\segoeprb.ttf",
                r"C:\Windows\Fonts\comicbd.ttf",
                r"C:\Windows\Fonts\seguisb.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            )
        else:
            names = (
                r"C:\Windows\Fonts\segoepr.ttf",
                r"C:\Windows\Fonts\comic.ttf",
                r"C:\Windows\Fonts\segoeui.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            )
        return names

    def _load_font(self, size: int, bold: bool) -> ImageFont.FreeTypeFont:
        for candidate in self._font_candidates(bold):
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _prepare_base(self, image: Image.Image) -> Image.Image:
        """
        Resize by height and crop away most of the transparent margin while
        preserving the irregular torn-paper edge on the left.
        """
        resized_width = max(1, round(image.width * self.height / image.height))
        resized = image.resize(
            (resized_width, self.height),
            Image.Resampling.LANCZOS,
        )

        alpha_bbox = resized.getchannel("A").getbbox()
        if alpha_bbox is None:
            natural_crop_x = 0
        else:
            natural_crop_x = max(0, alpha_bbox[0] - 6)

        # Reference crop used by the original 320 px sidebar.  Comparing it
        # with the widened crop gives the true on-panel x shift of the paper
        # artwork (and its built-in tabs/lines).  This keeps all runtime text
        # aligned with the PNG even after adding left overlap.
        reference_crop_x = natural_crop_x
        if reference_crop_x + self.content_width > resized.width:
            reference_crop_x = max(0, resized.width - self.content_width)

        crop_x = natural_crop_x
        if crop_x + self.width > resized.width:
            crop_x = max(0, resized.width - self.width)

        self.content_offset_x = max(0, reference_crop_x - crop_x)

        cropped = resized.crop(
            (crop_x, 0, crop_x + self.width, self.height)
        )

        if cropped.size != (self.width, self.height):
            canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            canvas.alpha_composite(cropped, (0, 0))
            return canvas

        return cropped

    def _bottom_paper_bounds(self, strip_height: int) -> tuple[int, int]:
        """Return visible paper bounds in the band covered by the strip."""
        band_top = max(0, self.height - max(1, int(strip_height)))
        alpha = np.asarray(self.base.getchannel("A"), dtype=np.uint8)
        band = alpha[band_top:self.height]

        # Ignore near-transparent anti-aliasing pixels at the torn edge.
        ys, xs = np.where(band >= 16)
        if xs.size == 0:
            return 0, self.width

        left = int(xs.min())
        right = int(xs.max()) + 1
        return left, right

    @staticmethod
    def _fit_inside(
        image: Image.Image,
        max_width: int,
        max_height: int,
    ) -> Image.Image:
        scale = min(
            max_width / image.width,
            max_height / image.height,
        )
        size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        return image.resize(size, Image.Resampling.LANCZOS)

    @classmethod
    def _trim_and_fit(
        cls,
        image: Image.Image,
        max_width: int,
        max_height: int,
        padding: int = 0,
    ) -> Image.Image:
        bbox = image.getchannel("A").getbbox()
        if bbox is not None:
            left = max(0, bbox[0] - padding)
            top = max(0, bbox[1] - padding)
            right = min(image.width, bbox[2] + padding)
            bottom = min(image.height, bbox[3] + padding)
            image = image.crop((left, top, right, bottom))
        return cls._fit_inside(image, max_width, max_height)

    @staticmethod
    def _safe_emotion(display_label: str) -> str:
        label = (display_label or "").strip().lower()
        return label if label in EMOTIONS else "neutral"

    @staticmethod
    def _confidence_text(confidence: Optional[float]) -> str:
        if confidence is None:
            return "--"
        value = max(0.0, min(1.0, float(confidence)))
        return f"{value * 100:.0f}%"

    @staticmethod
    def _draw_centered_text(
        draw: ImageDraw.ImageDraw,
        xy,
        text: str,
        font,
        fill,
    ) -> None:
        draw.text(xy, text, font=font, fill=fill, anchor="mm")

    @staticmethod
    def _draw_scribble_ring(
        draw: ImageDraw.ImageDraw,
        box,
        color,
    ) -> None:
        x1, y1, x2, y2 = box
        offsets = ((0, 0), (2, -1), (-2, 2))
        widths = (3, 2, 2)
        for (dx, dy), width in zip(offsets, widths):
            draw.ellipse(
                (x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                outline=color,
                width=width,
            )

    def render(
        self,
        display_label: str,
        confidence: Optional[float],
        preview_label: Optional[str],
        effects_enabled: bool,
        face_detected: bool,
        detected_label: str,
        raw_label: str,
        average_ms: float,
        debug_enabled: bool,
        display_fps: float = 0.0,
    ) -> np.ndarray:
        emotion = self._safe_emotion(display_label)
        accent_rgb = EMOTION_ACCENTS[emotion]
        accent = (*accent_rgb, 255)

        panel = self.base.copy()
        paper_offset_x = self.content_offset_x
        content_x = paper_offset_x + self.foreground_shift_x
        metrics_x = content_x + self.metrics_extra_shift_x
        metrics_y = self.metrics_y_shift

        # Bottom emotion strip sits behind the foreground content.
        strip = self.bottom_strips[emotion]
        panel.alpha_composite(
            strip,
            (self.bottom_strip_positions[emotion], self.height - strip.height),
        )

        # Emotion title.
        title = self.titles[emotion]
        title_x = content_x + (self.content_width - title.width) // 2 + 4
        title_y = 67 + max(0, (112 - title.height) // 2)
        panel.alpha_composite(title, (title_x, title_y))

        draw = ImageDraw.Draw(panel)

        # Confidence area.
        confidence_y_shift = -16
        ring_box = (70 + content_x, 202 + confidence_y_shift, 250 + content_x, 286 + confidence_y_shift)
        self._draw_scribble_ring(draw, ring_box, accent)
        self._draw_centered_text(
            draw,
            (160 + content_x, 244 + confidence_y_shift),
            self._confidence_text(confidence),
            self.font_percent,
            TEXT_DARK,
        )
        draw.text(
            (244 + content_x, 275 + confidence_y_shift),
            "confidence",
            font=self.font_small,
            fill=TEXT_DARK,
            anchor="lm",
        )

        # Runtime metrics between the two hand-drawn separators.
        tracking_value = "ON" if face_detected or preview_label is not None else "WAIT"
        speed_text = "--" if average_ms <= 0 else f"{average_ms:.1f} ms"
        rate_text = "--" if display_fps <= 0 else f"{display_fps:.1f} FPS"

        draw.ellipse(
            (40 + metrics_x, 337 + metrics_y, 49 + metrics_x, 346 + metrics_y),
            fill=accent if tracking_value == "ON" else TEXT_MUTED,
        )
        draw.text((55 + metrics_x, 333 + metrics_y), "TRACKING", font=self.font_small, fill=TEXT_DARK)
        draw.text((55 + metrics_x, 353 + metrics_y), tracking_value, font=self.font_regular, fill=TEXT_MUTED)

        draw.text((40 + metrics_x, 394 + metrics_y), "MODEL", font=self.font_small, fill=TEXT_DARK)
        draw.text((40 + metrics_x, 414 + metrics_y), "RTMPose", font=self.font_regular, fill=TEXT_MUTED)

        draw.text((184 + metrics_x, 333 + metrics_y), "SPEED", font=self.font_small, fill=TEXT_DARK)
        draw.text((184 + metrics_x, 353 + metrics_y), speed_text, font=self.font_regular, fill=TEXT_MUTED)

        draw.text((184 + metrics_x, 394 + metrics_y), "RATE", font=self.font_small, fill=TEXT_DARK)
        draw.text((184 + metrics_x, 414 + metrics_y), rate_text, font=self.font_regular, fill=TEXT_MUTED)

        # Labels on the two built-in paper tabs.
        mode_value = "Automatic" if preview_label is None else f"Preview {preview_label.title()}"
        effects_value = "On" if effects_enabled else "Off"

        self._draw_centered_text(
            draw,
            (106 + content_x + self.mode_label_shift_x, 518 + self.tab_label_y_shift),
            "MODE",
            self.font_small,
            TEXT_DARK,
        )
        self._draw_centered_text(
            draw,
            (232 + content_x + self.effects_label_shift_x, 518 + self.tab_label_y_shift),
            "EFFECTS",
            self.font_small,
            TEXT_DARK,
        )
        self._draw_centered_text(
            draw,
            (106 + content_x + self.mode_value_shift_x, 550),
            mode_value,
            self.font_tiny,
            TEXT_MUTED,
        )
        self._draw_centered_text(
            draw,
            (232 + content_x + self.effects_value_shift_x, 550),
            effects_value,
            self.font_regular,
            TEXT_MUTED,
        )

        # Bottom-right cloud sticker.
        cloud = self.clouds[emotion]
        cloud_x = self.width - cloud.width - self.cloud_right_inset
        cloud_y = (
            self.height
            - strip.height
            - cloud.height
            + self.cloud_bottom_strip_overlap
        )
        panel.alpha_composite(cloud, (cloud_x, cloud_y))

        # Keyboard help or detailed classifier state.
        draw = ImageDraw.Draw(panel)
        if debug_enabled:
            draw.text(
                (40 + content_x, self.shortcut_heading_y),
                "DETAILS",
                font=self.font_small,
                fill=TEXT_DARK,
            )
            debug_lines = (
                f"raw      {raw_label}",
                f"stable   {detected_label}",
                f"pred     {speed_text}",
                f"display  {rate_text}",
            )
            y = self.shortcut_start_y
            for line in debug_lines:
                draw.text((40 + content_x, y), line, font=self.font_tiny, fill=TEXT_MUTED)
                y += 17
        else:
            draw.text(
                (40 + content_x, self.shortcut_heading_y),
                "SHORTCUTS",
                font=self.font_small,
                fill=TEXT_DARK,
            )
            controls = (
                ("0", "Automatic"),
                ("1-7", "Preview"),
                ("E", "Effects"),
                ("D", "Details"),
                ("Q", "Quit"),
            )
            y = self.shortcut_start_y
            for key, action in controls:
                draw.text((40 + content_x, y), key, font=self.font_tiny, fill=TEXT_DARK)
                draw.text((75 + content_x, y), action, font=self.font_tiny, fill=TEXT_MUTED)
                y += self.shortcut_line_gap

        return np.asarray(panel, dtype=np.uint8)

    @staticmethod
    def alpha_blend_bgra_onto_bgr(
        background_bgr: np.ndarray,
        overlay_rgba: np.ndarray,
        x: int,
        y: int,
    ) -> None:
        """Alpha-composite an RGBA overlay onto a BGR image in place."""
        h, w = overlay_rgba.shape[:2]
        bg_h, bg_w = background_bgr.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + w)
        y2 = min(bg_h, y + h)
        if x1 >= x2 or y1 >= y2:
            return

        ox1 = x1 - x
        oy1 = y1 - y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        fg = overlay_rgba[oy1:oy2, ox1:ox2]
        fg_bgr = fg[..., :3][..., ::-1].astype(np.float32)
        alpha = (fg[..., 3:4].astype(np.float32) / 255.0)
        bg = background_bgr[y1:y2, x1:x2].astype(np.float32)

        blended = fg_bgr * alpha + bg * (1.0 - alpha)
        background_bgr[y1:y2, x1:x2] = blended.astype(np.uint8)