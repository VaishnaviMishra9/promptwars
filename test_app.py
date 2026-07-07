import sys
import unittest
from unittest.mock import MagicMock, patch

# --- Setup Mock Streamlit and dependencies ---
# This is necessary because app.py runs Streamlit code at the module level.
class MockSessionState(dict):
    """Mock Streamlit session state allowing dict and attribute access."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'MockSessionState' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __contains__(self, item):
        return super().__contains__(item)

# Create mock streamlit instance
mock_st = MagicMock()
mock_session_state = MockSessionState()
mock_st.session_state = mock_session_state

# Mock st.cache_resource decorator
def dummy_decorator(func):
    return func
mock_st.cache_resource = dummy_decorator

# Mock layout structures
mock_st.columns.return_value = [MagicMock() for _ in range(4)]
mock_st.sidebar = MagicMock()
mock_st.sidebar.radio.return_value = "🏟️ Fan Assistant"
mock_st.sidebar.text_input.return_value = ""
mock_st.sidebar.button.return_value = False

# Context Managers
class DummyContextManager:
    def __enter__(self):
        return MagicMock()
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

mock_st.chat_message.return_value = DummyContextManager()
mock_st.spinner.return_value = DummyContextManager()
mock_st.form.return_value = DummyContextManager()

# Mock st.form_submit_button at module level since app.py calls st.form_submit_button
mock_st.form_submit_button.return_value = False

# Inject mock streamlit into sys.modules
sys.modules['streamlit'] = mock_st

# Mock plotly to prevent drawing/rendering issues during import
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()

# Now import the app module under test
import app
import pandas as pd


class TestSmartStadiumApp(unittest.TestCase):
    """Unit test suite for verifying the FIFA 2026 Smart Stadium Assistant core logic."""

    def setUp(self):
        """Reset the mock session state before each test."""
        mock_session_state.clear()
        # Trigger the state initialization logic in app.py manually if cleared
        if 'chat_history' not in mock_session_state:
            mock_session_state.chat_history = [
                {"role": "assistant", "content": "Welcome to MetLife Stadium! I am your AI Fan Assistant for the FIFA World Cup 2026. How can I help you navigate the stadium, find concessions, or locate public transport today? ⚽"}
            ]
        if 'incidents' not in mock_session_state:
            mock_session_state.incidents = pd.DataFrame([
                {
                    "ID": "INC-001",
                    "Timestamp": "14:15",
                    "Category": "Facility",
                    "Location": "Sec 114, Row F, Seat 8",
                    "Severity": "Medium",
                    "Description": "Soda spilled near entrance, slip hazard.",
                    "Status": "Assigned",
                    "Reported By": "Volunteer #14"
                },
                {
                    "ID": "INC-002",
                    "Timestamp": "14:32",
                    "Category": "Technical",
                    "Location": "Gate C RFID Scanner 4",
                    "Severity": "High",
                    "Description": "Ticket scanner failed to connect. Fans forming a queue.",
                    "Status": "In Progress",
                    "Reported By": "Gate Staff #07"
                },
                {
                    "ID": "INC-003",
                    "Timestamp": "14:40",
                    "Category": "Medical",
                    "Location": "Concourse near Sec 218",
                    "Severity": "Critical",
                    "Description": "Fan experiencing heat-related dizziness, needs assistance.",
                    "Status": "Dispatched",
                    "Reported By": "First Aid Patrol 2"
                }
            ])
        if 'dashboard_metrics' not in mock_session_state:
            mock_session_state.dashboard_metrics = {
                "crowd_density": 88.5,
                "active_volunteers": 342,
                "clean_energy_pct": 76.8,
                "waste_diverted_pct": 69.4,
                "gates": {"Gate A": 120, "Gate B": 95, "Gate C": 240, "Gate D": 180},
                "transit_bus_wait": 15,
                "transit_train_wait": 6
            }

    def test_session_state_initialization(self):
        """Verify that standard session state objects initialize with correct types and keys."""
        self.assertIn('chat_history', mock_session_state)
        self.assertIsInstance(mock_session_state.chat_history, list)
        self.assertEqual(len(mock_session_state.chat_history), 1)
        self.assertEqual(mock_session_state.chat_history[0]["role"], "assistant")

        self.assertIn('incidents', mock_session_state)
        self.assertIsInstance(mock_session_state.incidents, pd.DataFrame)
        self.assertEqual(len(mock_session_state.incidents), 3)

        self.assertIn('dashboard_metrics', mock_session_state)
        metrics = mock_session_state.dashboard_metrics
        self.assertIsInstance(metrics, dict)
        required_keys = {
            "crowd_density", "active_volunteers", "clean_energy_pct",
            "waste_diverted_pct", "gates", "transit_bus_wait", "transit_train_wait"
        }
        self.assertTrue(required_keys.issubset(metrics.keys()))

    def test_incident_categories_and_schema(self):
        """Test incident logs structure, schema columns, and default categories."""
        df = mock_session_state.incidents
        expected_columns = [
            "ID", "Timestamp", "Category", "Location", "Severity", "Description", "Status", "Reported By"
        ]
        self.assertListEqual(list(df.columns), expected_columns)

        # Verify allowed categories check
        allowed_categories = {"Facility", "Medical", "Technical", "Safety/Security", "Other"}
        for category in df["Category"]:
            self.assertIn(category, allowed_categories)

    def test_refresh_dashboard_metrics(self):
        """Validate helper logic for updating and bound-checking dashboard metrics."""
        original_metrics = mock_session_state.dashboard_metrics.copy()
        
        # Invoke helper function to refresh metrics
        app.refresh_dashboard_metrics()
        
        updated_metrics = mock_session_state.dashboard_metrics
        
        # Verify values changed (since random numbers are added)
        self.assertNotEqual(original_metrics["crowd_density"], updated_metrics["crowd_density"])
        
        # Verify boundary constraints are maintained
        self.assertTrue(60.0 <= updated_metrics["crowd_density"] <= 100.0)
        self.assertTrue(updated_metrics["active_volunteers"] >= 200)
        self.assertTrue(50.0 <= updated_metrics["clean_energy_pct"] <= 100.0)
        self.assertTrue(50.0 <= updated_metrics["waste_diverted_pct"] <= 100.0)
        
        for gate, flow in updated_metrics["gates"].items():
            self.assertTrue(flow >= 20, f"{gate} entry rate dropped below minimum threshold.")

        self.assertTrue(updated_metrics["transit_bus_wait"] >= 2)
        self.assertTrue(updated_metrics["transit_train_wait"] >= 2)

    def test_get_gemini_client_invalid_or_missing_key(self):
        """Validate get_gemini_client returns None when API key is not provided."""
        self.assertIsNone(app.get_gemini_client(""))
        self.assertIsNone(app.get_gemini_client(None))

    @patch('app.genai.Client')
    def test_get_gemini_client_with_key(self, mock_client_class):
        """Verify get_gemini_client instantiates GenAI client when key is provided."""
        key = "AIzaSyFakeKey12345"
        app.get_gemini_client(key)
        mock_client_class.assert_called_once_with(api_key=key)

    def test_generate_chat_response_offline_transit(self):
        """Test offline simulated response matching transit queries."""
        # Ensure client is None to trigger simulated fallback response
        with patch('app.client', None):
            res = app.generate_chat_response("Tell me how to get to MetLife by transit")
            self.assertIn("Transportation Info", res)
            self.assertIn("NJ Transit Train", res)

    def test_generate_chat_response_offline_concessions(self):
        """Test offline simulated response matching concessions queries."""
        with patch('app.client', None):
            res = app.generate_chat_response("What concessions or food options are there?")
            self.assertIn("Concessions & Food Directory", res)
            self.assertIn("Street Tacos", res)

    def test_generate_chat_response_offline_ada(self):
        """Test offline simulated response matching ADA compliance queries."""
        with patch('app.client', None):
            res = app.generate_chat_response("Do you support ADA or wheelchair seating?")
            self.assertIn("ADA & Seating Access", res)
            self.assertIn("Verizon Gate", res)

    def test_generate_chat_response_offline_translation(self):
        """Test offline simulated response matching translation requests."""
        with patch('app.client', None):
            res = app.generate_chat_response("Translate emergency to Spanish")
            self.assertIn("Language Translation", res)
            self.assertIn("¿Dónde está la estación de primeros auxilios más cercana?", res)

    def test_generate_chat_response_offline_general(self):
        """Test offline simulated general response for other inputs."""
        with patch('app.client', None):
            res = app.generate_chat_response("Hello, what are the match details?")
            self.assertIn("Stadium General Guide", res)
            self.assertIn("offline simulated mode", res)

    @patch('app.types')
    def test_generate_chat_response_online(self, mock_types):
        """Test online response generation using the Google GenAI Client mock."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Welcome to the stadium! We have seats available in multiple zones."
        mock_client.models.generate_content.return_value = mock_response

        # Setup st.session_state.chat_history for the mock
        mock_session_state.chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        # Patch the global client object and call generate_chat_response
        with patch('app.client', mock_client):
            res = app.generate_chat_response("Are there open seats?")
            self.assertEqual(res, "Welcome to the stadium! We have seats available in multiple zones.")
            mock_client.models.generate_content.assert_called_once()


if __name__ == "__main__":
    unittest.main()
