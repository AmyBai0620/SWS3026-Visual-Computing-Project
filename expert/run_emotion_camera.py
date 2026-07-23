"""Canonical entry point for the final threaded scrapbook emotion camera."""

try:
    from realtime_demo_v10_threaded_scrapbook import base
except ImportError:
    from expert.realtime_demo_v10_threaded_scrapbook import base


if __name__ == "__main__":
    base.main()
