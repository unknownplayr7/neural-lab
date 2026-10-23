from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_starting_a_download():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
