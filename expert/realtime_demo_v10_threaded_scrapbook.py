from pathlib import Path

import numpy as np

try:
    import realtime_demo_v10_threaded as base
    from scrapbook_sidebar import ScrapbookSidebar
except ImportError:
    from expert import realtime_demo_v10_threaded as base
    from expert.scrapbook_sidebar import ScrapbookSidebar


BASE_DIR = Path(__file__).resolve().parent
UI_ASSET_ROOT = BASE_DIR / "assets" / "ui"

SIDEBAR_LEFT_OVERLAP = 72

sidebar = ScrapbookSidebar(
    asset_root=UI_ASSET_ROOT,
    width=base.SIDEBAR_WIDTH,
    height=base.APP_HEIGHT,
    left_overlap=SIDEBAR_LEFT_OVERLAP,
)


def compose_app_frame(
    camera_frame,
    detected_label: str,
    display_label: str,
    confidence,
    preview_label,
    average_ms: float,
    effects_enabled: bool,
    face_detected: bool,
    raw_label: str,
    debug_enabled: bool,
    display_fps: float = 0.0,
):
    """
    Compose v8f effects with a scrapbook sidebar that overlaps the
    camera on the left while keeping its right edge fixed.
    """
    # Use a quiet paper-coloured backdrop. We no longer stretch/copy the
    # camera's rightmost pixels underneath the transparent torn edge.
    canvas = np.full(
        (base.APP_HEIGHT, base.APP_WIDTH, 3),
        (238, 240, 242),
        dtype=np.uint8,
    )

    camera_view = base.fit_camera_view(
        camera_frame,
        base.VIDEO_WIDTH,
        base.APP_HEIGHT,
    )
    canvas[:, :base.VIDEO_WIDTH] = camera_view

    panel_rgba = sidebar.render(
        display_label=display_label,
        confidence=confidence,
        preview_label=preview_label,
        effects_enabled=effects_enabled,
        face_detected=face_detected,
        detected_label=detected_label,
        raw_label=raw_label,
        average_ms=average_ms,
        debug_enabled=debug_enabled,
        display_fps=display_fps,
    )

    # The panel is wider only on its left side. Anchoring it to the
    # application's right edge keeps the right boundary exactly unchanged.
    sidebar_x = base.APP_WIDTH - panel_rgba.shape[1]

    sidebar.alpha_blend_bgra_onto_bgr(
        canvas,
        panel_rgba,
        x=sidebar_x,
        y=0,
    )

    return canvas


base.WINDOW_NAME = "RTMPose Emotion Camera v10 - Threaded Scrapbook UI"
base.compose_app_frame = compose_app_frame


if __name__ == "__main__":
    base.main()
