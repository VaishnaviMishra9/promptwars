"""
FIFA World Cup 2026 Smart Stadium Operational Assistant
======================================================
This Streamlit-based web application provides an interactive control console
for MetLife Stadium matchday operations. It features:
1. A multilingual Fan Assistant chatbot leveraging the Google GenAI Gemini model.
2. A real-time Organizer Dashboard displaying telemetry visualizations and AI-powered operational assessment.
3. A real-time Incident Reporter console for logging and dispatcher scheduling.

Authors: VaishnaviMishra9 / promptwars
Version: 1.1.0 (Accessibility & Security Enhanced)
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import os
from google import genai
from google.genai import types
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. Page Configuration & Custom Theme Styling
# ==============================================================================
try:
    st.set_page_config(
        page_title="FIFA 2026 Smart Stadium Assistant",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    # Fail-safe print in case page config is initialized multiple times
    print(f"Page config error: {str(e)}")

# Inject custom CSS for premium design (glassmorphism, clean fonts, card layouts)
st.markdown("""
<style>
    /* Premium visual styling */
    .stApp {
        background-color: #0b0f19;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .main-title-container {
        background: linear-gradient(135deg, #10b981 0%, #1e3a8a 100%);
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15);
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(90deg, #ffd700, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .main-subtitle {
        color: #e2e8f0;
        font-size: 1.2rem;
        font-weight: 300;
    }
    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 215, 0, 0.3);
    }
    /* Dashboard section headings */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        border-left: 4px solid #10b981;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #f1f5f9;
    }
    /* Custom buttons styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. Stateful Data Store & Initialization
# ==============================================================================
# Initialize state tracking variables inside try-except block to handle memory/state faults safely.
try:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Welcome to MetLife Stadium! I am your AI Fan Assistant for the FIFA World Cup 2026. How can I help you navigate the stadium, find concessions, or locate public transport today? ⚽"
            }
        ]

    if 'incidents' not in st.session_state:
        st.session_state.incidents = pd.DataFrame([
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

    if 'dashboard_metrics' not in st.session_state:
        st.session_state.dashboard_metrics = {
            "crowd_density": 88.5,
            "active_volunteers": 342,
            "clean_energy_pct": 76.8,
            "waste_diverted_pct": 69.4,
            "gates": {"Gate A": 120, "Gate B": 95, "Gate C": 240, "Gate D": 180},
            "transit_bus_wait": 15,
            "transit_train_wait": 6
        }
except Exception as init_err:
    st.error("Operational Data System failed to initialize. Please try resetting the app.")
    st.stop()


def refresh_dashboard_metrics():
    """
    Safely updates and randomizes stadium dashboard telemetry metrics.
    Ensures that updated values remain within logical operational limits
    (e.g., crowd density between 60% and 100%, and wait times above 2 minutes).
    """
    try:
        # Create a shallow copy of existing metrics to prevent state mutation errors
        metrics = st.session_state.dashboard_metrics.copy()
        
        # Apply slight random deviations to simulate dynamic sensor metrics
        metrics["crowd_density"] = min(100.0, max(60.0, metrics["crowd_density"] + random.uniform(-2.5, 2.5)))
        metrics["active_volunteers"] = int(max(200, metrics["active_volunteers"] + random.randint(-5, 8)))
        metrics["clean_energy_pct"] = min(100.0, max(50.0, metrics["clean_energy_pct"] + random.uniform(-1.5, 1.5)))
        metrics["waste_diverted_pct"] = min(100.0, max(50.0, metrics["waste_diverted_pct"] + random.uniform(-0.8, 0.8)))
        
        # Gate entry flows (fans per minute)
        metrics["gates"] = {
            "Gate A": max(20, metrics["gates"]["Gate A"] + random.randint(-15, 15)),
            "Gate B": max(20, metrics["gates"]["Gate B"] + random.randint(-10, 10)),
            "Gate C": max(20, metrics["gates"]["Gate C"] + random.randint(-25, 25)),
            "Gate D": max(20, metrics["gates"]["Gate D"] + random.randint(-20, 20))
        }
        
        # Public Transit queue/wait updates
        metrics["transit_bus_wait"] = max(2, metrics["transit_bus_wait"] + random.randint(-3, 3))
        metrics["transit_train_wait"] = max(2, metrics["transit_train_wait"] + random.randint(-1, 1))
        
        # Write back to session state securely
        st.session_state.dashboard_metrics = metrics
    except KeyError as key_err:
        st.error(f"Error accessing telemetry metrics: {str(key_err)}. Defaulting to placeholder values.")
    except Exception as e:
        st.error("Failed to update telemetry metrics. Restoring previous state.")

# ==============================================================================
# 3. Sidebar Navigation & API Setup
# ==============================================================================
st.sidebar.markdown(
    """
    <div class="sidebar-logo">
        <h2 style="margin: 0; font-size: 1.3rem; font-weight: bold; color: white; text-align: center;">🏆 SMART STADIUM</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="background-color: rgba(255,255,255,0.04); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #ffd700; font-size: 1.05rem; text-align: center;">🏟️ METLIFE STADIUM</h4>
        <p style="margin: 4px 0; font-size: 0.85rem; color: #94a3b8; text-align: center;">East Rutherford, NY/NJ</p>
        <p style="margin: 4px 0; font-size: 0.85rem; color: #94a3b8; text-align: center;">📅 Matchday 4 • June 18, 2026</p>
        <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.8rem; color: #94a3b8;">Status</span>
            <span style="background-color: #10b981; color: white; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: bold;">ONLINE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar Radio navigation with Accessibility label and description help
navigation = st.sidebar.radio(
    "SELECT APPLICATION MODE",
    ["🏟️ Fan Assistant", "📊 Organizer Dashboard", "🚨 Incident Reporter"],
    index=0,
    help="Select which module of the Smart Stadium Assistant you would like to open."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Gemini API Configuration")

# Secure API Key input with custom labels and help tooltip
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    placeholder="Enter API Key here...",
    help="Input your Google Gemini API Key. If left empty, the application will default to offline simulation mode."
)

# Display connection status
if api_key_input:
    st.sidebar.success("Gemini API Key Loaded!")
else:
    st.sidebar.warning("Running in Simulated (Offline) Mode.")

# Sidebar telemetry updates
st.sidebar.markdown("---")
if st.sidebar.button(
    "🔄 Refresh Stadium Metrics",
    help="Triggers a simulation to refresh the stadium's active telemetry data."
):
    refresh_dashboard_metrics()
    st.sidebar.info("Dashboard telemetry updated!")


@st.cache_resource
def get_gemini_client(api_key):
    """
    Initialize the Google GenAI Client with the user-provided API key.

    Args:
        api_key (str): The Gemini API credential string.

    Returns:
        genai.Client or None: Instantiated client object if successful, else None.
    """
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as connection_err:
        st.sidebar.error("Failed to authenticate/connect to Google GenAI API. Reverting to offline mode.")
        return None

# Instantiating client (returns None if no key configured)
client = get_gemini_client(api_key_input)

# ==============================================================================
# 4. Main Page Header
# ==============================================================================
st.markdown(
    """
    <div class="main-title-container">
        <div class="main-title">FIFA WORLD CUP 2026</div>
        <div class="main-subtitle">🏟️ Smart Stadium Operational Assistant • Control Console</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 5. Page 1: Fan Assistant
# ==============================================================================
if navigation == "🏟️ Fan Assistant":
    st.header("💬 Multilingual Fan Assistant")
    st.write(
        "Welcome to the FIFA World Cup 2026 stadium companion! Use this AI assistant to ask questions "
        "about seating maps, transit coordinates, amenities, or to translate key sentences into multiple languages."
    )
    
    # Language selection dropdown (Annotated with accessibility attributes)
    language = st.selectbox(
        "Select your preferred language / Selecciona tu idioma / Choisissez votre langue",
        ["English 🇺🇸", "Español 🇪🇸", "Français 🇫🇷", "Português 🇧🇷", "Deutsch 🇩🇪", "العربية 🇸🇦"],
        index=0,
        help="Select the language you want the AI assistant to reply in."
    )
    
    # Suggested Prompts section
    st.markdown("##### 💡 Suggested Questions")
    col1, col2, col3, col4 = st.columns(4)
    
    suggested_q = None
    with col1:
        if st.button("📍 Transit to MetLife", help="Click to load routing instructions to MetLife Stadium."):
            suggested_q = "How do I get to MetLife Stadium by public transit from Manhattan?"
    with col2:
        if st.button("🍕 Concessions Guide", help="Click to view food options and vegan choices available in the stadium."):
            suggested_q = "Where are the concessions located, and what vegan options are available?"
    with col3:
        if st.button("♿ Seating & ADA Access", help="Click to ask about elevator locations and accessible entrance gates."):
            suggested_q = "I have an ADA ticket. Which gate should I enter, and where are the escalators?"
    with col4:
        if st.button("🌎 Translate to Spanish", help="Click to check translation for safety-related questions."):
            suggested_q = "Translate 'Where is the nearest first aid station?' into Spanish and provide pronunciation."

    def generate_chat_response(prompt_text):
        """
        Processes chat prompts from the user. Integrates with the Gemini API if
        a client is online, or uses a local keywords router for simulated responses.

        Args:
            prompt_text (str): Question or message submitted by the fan.

        Returns:
            str: AI-generated or simulated text answer.
        """
        system_instruction = (
            f"You are the FIFA World Cup 2026 Smart Stadium Assistant. You are currently helping a fan "
            f"in their preferred language: {language}. "
            f"Your persona is a friendly, professional, enthusiastic, and highly knowledgeable stadium guide. "
            f"You assist fans with stadium seating, navigation, food concessions, amenities, security guidelines, "
            f"and public transit FAQs. Keep your responses crisp, informative, and formatted with bullet points where appropriate."
        )

        if not client:
            # Fallback simulated response handling
            try:
                time.sleep(1)  # Simulate API processing latency
                prompt_lower = prompt_text.lower()
                
                if "transit" in prompt_lower or "get to" in prompt_lower or "manhattan" in prompt_lower:
                    return (
                        f"🚇 **[Simulated Mode] Transportation Info:**\n\n"
                        f"To get to MetLife Stadium from Manhattan:\n"
                        f"1. **NJ Transit Train:** Take the train from Penn Station (NY) to Secaucus Junction. Transfer to the Meadowlands Rail Line direct to the stadium. Service runs every 10 minutes on match days.\n"
                        f"2. **Express Bus (Route 351):** Departs from Port Authority Bus Terminal starting 3.5 hours before kickoff, returning up to 1 hour after the match.\n"
                        f"3. **Ride-Share:** Drop-off and pick-up zones are located in **Lot G** and **Lot E**."
                    )
                elif "concession" in prompt_lower or "vegan" in prompt_lower or "food" in prompt_lower:
                    return (
                        f"🌭 **[Simulated Mode] Concessions & Food Directory:**\n\n"
                        f"MetLife Stadium has a diverse selection of food options:\n"
                        f"- **Section 117 & 142:** Street Tacos, Nachos, and Quesadillas 🌮\n"
                        f"- **Section 104:** Green Greens Salad Bar (Vegan & Gluten-free choices available) 🥗\n"
                        f"- **Section 124 & 224:** Craft Burgers, Classic Stadium Hot Dogs, and Fries 🍔\n"
                        f"- **Section 134:** Halal cart items & Gyros 🥙"
                    )
                elif "ada" in prompt_lower or "seat" in prompt_lower or "escalator" in prompt_lower:
                    return (
                        f"♿ **[Simulated Mode] ADA & Seating Access:**\n\n"
                        f"MetLife Stadium is fully accessible:\n"
                        f"- **Drop-off:** ADA shuttle service is available from all parking lots. Accessible drop-off is near Gate C.\n"
                        f"- **Entrance:** Guests with disabilities can enter through any gate. The **Verizon Gate** (East side) has the shortest path to concourse elevators.\n"
                        f"- **Elevators:** Located at the Verizon Gate, MetLife Gate, and Pepsi Gate. Escalators are located at all primary gate entrances."
                    )
                elif "translate" in prompt_lower or "spanish" in prompt_lower:
                    return (
                        f"🌎 **[Simulated Mode] Language Translation:**\n\n"
                        f"**English:** \"Where is the nearest first aid station?\"\n"
                        f"**Spanish:** \"¿Dónde está la estación de primeros auxilios más cercana?\"\n"
                        f"**Pronunciation Guide:** *Dohn-deh ehs-tah lah ehs-tah-see-ohn deh pree-meh-rohs ah-ooh-see-lee-ohs mahs sehr-cah-nah?*"
                    )
                else:
                    return (
                        f"🤖 **[Simulated Mode] Stadium General Guide:**\n\n"
                        f"Thank you for reaching out! Since there is no active Gemini API key configured, "
                        f"I'm operating in offline simulated mode. I can respond to topics regarding Transit, "
                        f"Concessions, ADA seating, and translations.\n\n"
                        f"To unlock unrestricted conversational AI capabilities, please input a valid Google Gemini API Key in the sidebar."
                    )
            except Exception as sim_err:
                return "⚠️ A local system error occurred while generating simulated instructions. Please try again."

        # Online response generation via Gemini API
        try:
            # Prepare sliding chat history frame for the GenAI SDK
            contents = []
            for msg in st.session_state.chat_history[-6:]:
                contents.append(types.Content(
                    role="user" if msg["role"] == "user" else "model",
                    parts=[types.Part.from_text(text=msg["content"])]
                ))
            
            # Append prompt if not already present
            if not contents or contents[-1].parts[0].text != prompt_text:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt_text)]
                ))

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=800
            )
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=config
            )
            return response.text
        except Exception as api_err:
            return f"⚠️ Connection/API Error: {str(api_err)}. Please verify your network and Gemini API key status."

    # Render Chat History
    chat_container = st.container()
    with chat_container:
        try:
            for message in st.session_state.chat_history:
                role_emoji = "🤖" if message["role"] == "assistant" else "👤"
                with st.chat_message(message["role"]):
                    st.markdown(f"**{role_emoji} {message['role'].title()}**")
                    st.markdown(message["content"])
        except Exception as render_err:
            st.error("Could not load full conversation layout. Re-initializing view...")

    # Action routing if suggested prompt is triggered
    if suggested_q:
        try:
            st.session_state.chat_history.append({"role": "user", "content": suggested_q})
            with st.chat_message("user"):
                st.markdown(f"**👤 User**")
                st.markdown(suggested_q)
            
            with st.spinner("AI Assistant is typing..."):
                ai_response = generate_chat_response(suggested_q)
            
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(f"**🤖 Assistant**")
                st.markdown(ai_response)
            st.rerun()
        except Exception as action_err:
            st.error("Error executing prompt redirect. Please refresh and try again.")

    # Chat Input block with clear placeholder for keyboard accessibility
    if user_input := st.chat_input("Ask a question (e.g., 'What is the bag policy?')"):
        try:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(f"**👤 User**")
                st.markdown(user_input)
                
            with st.spinner("AI Assistant is typing..."):
                ai_response = generate_chat_response(user_input)
                
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(f"**🤖 Assistant**")
                st.markdown(ai_response)
            st.rerun()
        except Exception as chat_err:
            st.error("Failed to post message. Please try sending again.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History", help="Delete current chat logs and restart conversation."):
        try:
            st.session_state.chat_history = [
                {
                    "role": "assistant", 
                    "content": "Welcome to MetLife Stadium! I am your AI Fan Assistant for the FIFA World Cup 2026. How can I help you today? ⚽"
                }
            ]
            st.rerun()
        except Exception as clear_err:
            st.error("Error resetting chat logs.")

# ==============================================================================
# 6. Page 2: Organizer Dashboard
# ==============================================================================
elif navigation == "📊 Organizer Dashboard":
    st.header("📊 Organizer Operational Intelligence Center")
    st.write(
        "A real-time telemetry control center for stadium management. Track crowd dynamics, "
        "energy grids, waste diversion indexes, and utilize AI Decision Support to resolve operations anomalies."
    )
    
    # Retrieve metrics safely
    try:
        m = st.session_state.dashboard_metrics
        active_incidents_count = len(st.session_state.incidents)
    except Exception as data_ret_err:
        st.error("Unable to load active telemetry datasets.")
        st.stop()
    
    # Layout - Key Metrics Cards Grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">👥 Crowd Density</div>
                <div class="metric-value">{m['crowd_density']:.1f}%</div>
                <div style="font-size: 0.85rem; color: {'#ef4444' if m['crowd_density'] > 90 else '#10b981'}; font-weight: bold;">
                    {'⚠️ CRITICAL FILL' if m['crowd_density'] > 90 else '✓ OPTIMAL'}
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🚨 Active Incidents</div>
                <div class="metric-value">{active_incidents_count}</div>
                <div style="font-size: 0.85rem; color: {'#ef4444' if active_incidents_count > 0 else '#10b981'}; font-weight: bold;">
                    {'⚠️ RESPONSE REQUIRED' if active_incidents_count > 0 else '✓ CLEAR'}
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🌱 Green Energy Share</div>
                <div class="metric-value">{m['clean_energy_pct']:.1f}%</div>
                <div style="font-size: 0.85rem; color: #10b981; font-weight: bold;">
                    ⚡ Clean Solar Grid Active
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">♻️ Waste Diversion</div>
                <div class="metric-value">{m['waste_diverted_pct']:.1f}%</div>
                <div style="font-size: 0.85rem; color: #10b981; font-weight: bold;">
                    ♻️ Sorting Targets Met
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        
    st.markdown("<div class='section-header'>Live Telemetry Visualization</div>", unsafe_allow_html=True)
    
    # Telemetry Plots Grid
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        try:
            # Plot 1: Gate flow rate
            gates_list = list(m["gates"].keys())
            flows_list = list(m["gates"].values())
            
            df_gates = pd.DataFrame({
                "Gate": gates_list,
                "Fans Entry Rate (fans/min)": flows_list
            })
            
            # Color gates dynamic thresholding
            colors = []
            for v in flows_list:
                if v > 200:
                    colors.append('#ef4444')  # Red warning
                elif v > 150:
                    colors.append('#f59e0b')  # Amber
                else:
                    colors.append('#10b981')  # Green
                    
            fig_gates = px.bar(
                df_gates, x="Gate", y="Fans Entry Rate (fans/min)",
                title="Real-time Entrance Gate Flow Rates",
                text="Fans Entry Rate (fans/min)",
                color="Gate",
                color_discrete_map={"Gate A": colors[0], "Gate B": colors[1], "Gate C": colors[2], "Gate D": colors[3]}
            )
            fig_gates.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_gates, use_container_width=True)
        except Exception as plot1_err:
            st.warning("Failed to render Gate Flow telemetry chart. Displaying raw data instead.")
            st.write(df_gates if 'df_gates' in locals() else m["gates"])
        
    with plot_col2:
        try:
            # Plot 2: Waste diversion components
            df_waste = pd.DataFrame({
                "Category": ["Compostable Organics", "Recyclable Plastics", "Landfill Waste"],
                "Weight (Tons)": [14.2, 9.8, 5.1]
            })
            fig_waste = px.pie(
                df_waste, values="Weight (Tons)", names="Category", hole=0.6,
                title="Operational Waste Diversion Metrics",
                color_discrete_sequence=["#10b981", "#3b82f6", "#ef4444"]
            )
            fig_waste.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_waste, use_container_width=True)
        except Exception as plot2_err:
            st.warning("Failed to render Waste metrics chart.")
            st.write(df_waste if 'df_waste' in locals() else "Diversion rate: " + str(m['waste_diverted_pct']))

    # Grid details (Transit and volunteer counts)
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.subheader("🚌 Public Transit Status")
        st.write(f"- 🚌 **Bus Terminal Line Wait:** Approx. `{m['transit_bus_wait']}` minutes.")
        st.write(f"- 🚇 **Meadowlands Train Line Departure:** Approx. every `{m['transit_train_wait']}` minutes.")
    with detail_col2:
        st.subheader("👷 Personnel Metrics")
        st.write(f"- 📋 **Active Volunteers On-Duty:** `{m['active_volunteers']}` staff members.")
        st.write(f"- 🚨 **Incident Dispatch Readiness:** `{max(5, int(m['active_volunteers'] * 0.05))}` standby teams.")

    # AI Decision Support Panel
    st.markdown("<div class='section-header'>🤖 AI Operational Intelligence Dispatcher</div>", unsafe_allow_html=True)
    st.write("Generate a real-time event management report based on current crowd sensors and logs.")
    
    # Run Operational Analysis (Adding clear description and accessibility tooltip)
    if st.button(
        "⚡ Run AI Operations Analysis",
        help="Analyzes current telemetry metrics and incident logs to generate an actionable director's brief."
    ):
        with st.spinner("Analyzing live feeds and log registers..."):
            try:
                # Prepare analysis briefing prompt
                incidents_text = ""
                if st.session_state.incidents.empty:
                    incidents_text = "No active incidents."
                else:
                    for _, row in st.session_state.incidents.iterrows():
                        incidents_text += f"- [{row['Severity']}] {row['Category']} at {row['Location']}: {row['Description']} ({row['Status']})\n"
                
                prompt = (
                    f"You are the Lead Event Operations Director for FIFA World Cup 2026. Provide an operational analysis "
                    f"and key action points based on the current live stadium metrics:\n\n"
                    f"**Live Metrics:**\n"
                    f"- Crowd Density: {m['crowd_density']:.1f}% capacity\n"
                    f"- Gate Flows (fans/min): {m['gates']}\n"
                    f"- Energy Grid: Renewable share: {m['clean_energy_pct']:.1f}%\n"
                    f"- Waste Diversion Rate: {m['waste_diverted_pct']:.1f}%\n"
                    f"- Transit Wait Times: Bus: {m['transit_bus_wait']}m, Train: {m['transit_train_wait']}m\n\n"
                    f"**Active Incidents:**\n"
                    f"{incidents_text}\n\n"
                    f"Generate a concise operational assessment report under the following headings:\n"
                    f"1. 🏟️ **Crowd Flow & Transit Management**\n"
                    f"2. 🌿 **Sustainability & Grid Efficiency**\n"
                    f"3. 🚨 **Incident Triage & Security Briefing**\n"
                    f"Keep the tone urgent, professional, and tactical. Use bullet points."
                )
                
                if not client:
                    # Offline simulated briefing
                    time.sleep(1.5)
                    ai_briefing = (
                        f"### 📋 **FIFA Match Operations Briefing [Simulated Mode]**\n\n"
                        f"#### 🏟️ **Crowd Flow & Transit Management**\n"
                        f"- **Gate Congestion:** Gate C is experiencing elevated flow rate ({m['gates']['Gate C']} fans/min). Direct stadium monitors to redeploy crowd control rope barriers. Recommend broadcasting PA announcements advising incoming crowds to use Gate B.\n"
                        f"- **Transit Operations:** Train departures are optimal. Bus line wait times ({m['transit_bus_wait']}m) are within normal margins. Deploy local transit marshals to Lot G to manage exit queues.\n\n"
                        f"#### 🌿 **Sustainability & Grid Efficiency**\n"
                        f"- **Energy Balance:** Solar grid input is steady at {m['clean_energy_pct']:.1f}%. The stadium HVAC system is optimized. No high-draw exceptions detected.\n"
                        f"- **Composting Targets:** Diversion rate is {m['waste_diverted_pct']:.1f}%. Ensure volunteers replace overflow recycling bags in Concourse Zones 3 & 4.\n\n"
                        f"#### 🚨 **Incident Triage & Security Briefing**\n"
                        f"- **Urgent Medical Response:** Triage unit must expedite dispatch to Concourse Sec 218 for the dizzy spectator. Confirm paramedic transit clearance.\n"
                        f"- **Scanner Failure:** Technical crew must reboot Gate C Scanner 4. Keep manual barcode app terminals active on perimeter standby."
                    )
                else:
                    try:
                        config = types.GenerateContentConfig(
                            temperature=0.3,
                            max_output_tokens=900
                        )
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=prompt,
                            config=config
                        )
                        ai_briefing = response.text
                    except Exception as api_err:
                        ai_briefing = f"⚠️ API Error performing analysis: {str(api_err)}"
                        
                st.markdown(
                    f"""
                    <div style="background-color: rgba(16, 185, 129, 0.05); padding: 25px; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.2); box-shadow: inset 0 0 10px rgba(16, 185, 129, 0.05);">
                        <h4 style="margin-top: 0; color: #10b981;">📋 Real-time AI Operations Assessment</h4>
                        <hr style="border: 0; border-top: 1px solid rgba(16, 185, 129, 0.2); margin: 12px 0;">
                        {ai_briefing}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error("Failed to generate AI operational assessment due to an unexpected system error.")

# ==============================================================================
# 7. Page 3: Real-time Incident Reporter
# ==============================================================================
elif navigation == "🚨 Incident Reporter":
    st.header("🚨 Real-time Incident Reporter Console")
    st.write(
        "For stadium operations staff, security coordinators, and field volunteers. "
        "Log new issues instantly and receive custom AI-generated dispatch guidelines."
    )
    
    col_form, col_log = st.columns([1, 1.2])
    
    with col_form:
        st.subheader("📝 Report New Incident")
        # Secure incident reporting inputs wrapped inside try-except
        try:
            with st.form("new_incident_form", clear_on_submit=True):
                category = st.selectbox(
                    "Category",
                    ["Facility", "Medical", "Technical", "Safety/Security", "Other"],
                    help="Select the category that best fits the nature of the issue."
                )
                location = st.text_input(
                    "Location / Seating Section",
                    placeholder="e.g. Sec 102, Row B, Seat 4",
                    help="Describe where the incident occurred. Include section, row, and seat values."
                )
                severity = st.select_slider(
                    "Severity Level",
                    options=["Low", "Medium", "High", "Critical"],
                    help="Choose the triage tier to denote immediate response urgency."
                )
                description = st.text_area(
                    "Issue Description",
                    placeholder="Provide detail (e.g., wet floor, broken scanner, fan injury)...",
                    help="Explain details of the problem so dispatch teams bring correct tools/gear."
                )
                reporter = st.text_input(
                    "Reporter ID / Unit",
                    placeholder="e.g. Volunteer #18, Usher Unit C",
                    help="Enter your staff/volunteer ID or leave blank to report anonymously."
                )
                
                # Corrected form submit button to avoid Streamlit runtime errors, added help descriptor
                submitted = st.form_submit_button(
                    "🚨 Submit Incident Report",
                    help="Submit this form to add the incident log to the operational datastore."
                )
                
                if submitted:
                    if not location or not description:
                        st.error("Please fill in the Location and Issue Description fields.")
                    else:
                        try:
                            # Generate ID
                            new_id = f"INC-{random.randint(100, 999)}"
                            current_time = time.strftime("%H:%M")
                            
                            new_row = {
                                "ID": new_id,
                                "Timestamp": current_time,
                                "Category": category,
                                "Location": location,
                                "Severity": severity,
                                "Description": description,
                                "Status": "Submitted",
                                "Reported By": reporter if reporter else "Anonymous"
                            }
                            
                            # Concatenate pandas rows safely
                            st.session_state.incidents = pd.concat([
                                st.session_state.incidents,
                                pd.DataFrame([new_row])
                            ], ignore_index=True)
                            st.success(f"Success: Incident {new_id} logged successfully!")
                        except Exception as log_err:
                            st.error(f"Failed to record new report to local state: {str(log_err)}")
        except Exception as form_err:
            st.error("Incident submission form failed to render. Please reload and try again.")
                    
    with col_log:
        st.subheader("📋 Active Incident Log")
        
        try:
            if st.session_state.incidents.empty:
                st.info("No active incidents logged.")
            else:
                # Custom styled dataframe view
                st.dataframe(
                    st.session_state.incidents,
                    column_config={
                        "Severity": st.column_config.TextColumn("Severity"),
                        "Status": st.column_config.SelectboxColumn("Status", options=["Submitted", "Assigned", "In Progress", "Dispatched", "Resolved"])
                    },
                    use_container_width=True,
                    hide_index=True
                )
        except Exception as table_err:
            st.error("Error reading active incident database log.")
            
        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 15px 0;'>", unsafe_allow_html=True)
        
        # Select Incident for Dispatch Guidelines
        st.subheader("🤖 AI Dispatch Protocol Planner")
        
        try:
            incident_ids = st.session_state.incidents["ID"].tolist() if not st.session_state.incidents.empty else []
            selected_incident_id = st.selectbox(
                "Select Incident for AI Action Plan",
                options=incident_ids,
                help="Select one of the logged incidents from the dropdown to formulate a dispatch response plan."
            )
            
            if selected_incident_id:
                incident_row = st.session_state.incidents[st.session_state.incidents["ID"] == selected_incident_id].iloc[0]
                
                # Button to generate dispatch plan
                if st.button(
                    "⚡ Generate Dispatch Protocol",
                    help="Queries Gemini to generate emergency dispatch coordinates, gear requirements, and safety procedures."
                ):
                    with st.spinner(f"Analyzing SOPs for {selected_incident_id}..."):
                        try:
                            prompt = (
                                f"You are the FIFA Stadium Safety & Security Dispatch Coordinator. Generate a step-by-step dispatch action "
                                f"plan for the following reported incident:\n\n"
                                f"**Incident ID:** {incident_row['ID']}\n"
                                f"**Category:** {incident_row['Category']}\n"
                                f"**Severity:** {incident_row['Severity']}\n"
                                f"**Location:** {incident_row['Location']}\n"
                                f"**Description:** {incident_row['Description']}\n\n"
                                f"Provide a structured response plan detailing:\n"
                                f"1. **Triage Priority** (Immediate threat level and urgency rating)\n"
                                f"2. **Dispatch Actions** (Who to send, what gear is needed)\n"
                                f"3. **Standard Operating Procedure (SOP) Reference** (Relevant stadium policy for this issue)\n"
                                f"4. **On-Scene Instructions for Staff/Volunteers** (What the reporting volunteer should do until help arrives)\n\n"
                                f"Make it brief, actionable, and formatted cleanly using markdown card style."
                            )
                            
                            if not client:
                                # Offline simulated response
                                time.sleep(1)
                                action_plan = (
                                    f"### 📋 **Response Protocol for {incident_row['ID']} [Simulated]**\n\n"
                                    f"- **Triage Priority:** **{incident_row['Severity'].upper()}** - Action required within 5-10 minutes.\n"
                                    f"- **Dispatch Action:** Notify the nearest zone coordinator. Send a response unit "
                                    f"({ 'First-Aid Crew' if incident_row['Category'] == 'Medical' else 'Facility Janitorial Team' if incident_row['Category'] == 'Facility' else 'IT Scanner Technical Lead' if incident_row['Category'] == 'Technical' else 'Security Unit Patrol' }) to {incident_row['Location']}.\n"
                                    f"- **SOP Protocol Reference:** Section 14.2 (Facility Hazards) or Section 8.1 (Emergency Care). Standard response is to verify threat, secure perimeter, and clear lines for dispatcher.\n"
                                    f"- **Reporter Guidance:** Remain at location, ensure safety of nearby fans, and notify responder when they arrive."
                                )
                            else:
                                try:
                                    config = types.GenerateContentConfig(
                                        temperature=0.2,
                                        max_output_tokens=700
                                    )
                                    response = client.models.generate_content(
                                        model="gemini-2.5-flash",
                                        contents=prompt,
                                        config=config
                                    )
                                    action_plan = response.text
                                except Exception as api_err:
                                    action_plan = f"⚠️ API Error generating action plan: {str(api_err)}"
                                    
                            st.markdown(
                                f"""
                                <div style="background-color: rgba(239, 68, 68, 0.05); padding: 20px; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.2);">
                                    <h4 style="margin-top: 0; color: #ef4444;">🛡️ Dispatch Protocol Briefing ({selected_incident_id})</h4>
                                    <hr style="border: 0; border-top: 1px solid rgba(239, 68, 68, 0.2); margin: 10px 0;">
                                    {action_plan}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.error("Error analyzing incident parameters. Failed to run dispatch generation.")
        except Exception as id_err:
            st.error("Could not construct incident list dropdown.")

        # Clear incidents database
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Incident Log", help="Wipes all logged entries in the active incident log table."):
            try:
                st.session_state.incidents = pd.DataFrame(columns=[
                    "ID", "Timestamp", "Category", "Location", "Severity", "Description", "Status", "Reported By"
                ])
                st.rerun()
            except Exception as clear_err:
                st.error("Failed to wipe incident datastore.")

# ==============================================================================
# 8. Footer Section
# ==============================================================================
st.markdown("<br><hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.05); margin: 30px 0 10px 0;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; color: #64748b; font-size: 0.8rem; padding: 0 10px;">
        <span>FIFA World Cup 2026 Smart Stadium Assistant Console</span>
        <span>Version 1.1.0 (Pre-match build)</span>
    </div>
    """,
    unsafe_allow_html=True
)
