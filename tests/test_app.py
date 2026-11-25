import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)

    def test_initial_view_and_parameter_changes(self):
        app = self.app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 4)
        self.assertEqual(len(app.dataframe), 2)
        app.selectbox(key="dim").select(64)
        app.slider(key="offset").set_value(100)
        app.checkbox(key="causal").check().run()
        app.selectbox(key="pair").select(31).run()
        self.assertEqual(len(app.exception), 0)
        app.selectbox(key="dim").select(8).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn(app.selectbox(key="pair").value, range(4))

    def test_empty_long_and_single_token_input(self):
        app = self.app.run()
        app.text_area(key="text").input("  ").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.info), 1)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        app.text_area(key="text").input("gato " * 49).run()
        self.assertIn("48", app.info[0].value)
        app.text_area(key="text").input("ação").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 4)


if __name__ == "__main__":
    unittest.main()
