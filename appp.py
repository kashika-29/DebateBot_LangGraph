"""
DebateBot: Two Sides and a Judge
AI-powered debate arena with LangGraph, Groq, and Tavily
"""
import random
from datetime import datetime
import streamlit as st
import os
import json
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from functools import lru_cache

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Initialize API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in .env file.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY not found in environment variables. Please set it in .env file.")
    st.stop()


# ============================================================================
# LANGGRAPH AGENT SYSTEM
# ============================================================================

class DebateState(TypedDict):
    """State for the debate graph"""
    topic: str
    current_round: int
    total_rounds: int
    pro_messages: Sequence[BaseMessage]
    con_messages: Sequence[BaseMessage]
    pro_score: float
    con_score: float
    citations: List[Dict[str, Any]]
    debate_complete: bool


@lru_cache(maxsize=1)
def get_pro_llm():
    """Get cached Pro LLM instance"""
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.7,
        api_key=GROQ_API_KEY,
        streaming=True
    )

def get_pro_prompt(length: str = 'medium'):
    """Get Pro prompt template with dynamic length"""
    length_instructions = {
        'short': 'Keep your argument concise (50-80 words, 3-5 sentences).',
        'medium': 'Provide a well-developed argument (100-150 words) with explanation and one supporting example.',
        'long': 'Provide a detailed argument (180-250 words) with multiple supporting points, counterarguments where appropriate, and a strong conclusion.'
    }
    
    instruction = length_instructions.get(length, length_instructions['medium'])
    system_prompt = f"""Argue IN FAVOR of the topic. {instruction} Never identify as "pro agent" - just argue directly."""
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

def create_pro_agent(length: str = 'medium'):
    """Create the Pro agent with Groq LLM"""
    return get_pro_prompt(length) | get_pro_llm()


@lru_cache(maxsize=1)
def get_con_llm():
    """Get cached Con LLM instance"""
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.7,
        api_key=GROQ_API_KEY,
        streaming=True
    )

def get_con_prompt(length: str = 'medium'):
    """Get Con prompt template with dynamic length"""
    length_instructions = {
        'short': 'Keep your rebuttal concise (50-80 words, 3-5 sentences).',
        'medium': 'Provide a well-developed rebuttal (100-150 words) with explanation and one supporting example.',
        'long': 'Provide a detailed rebuttal (180-250 words) with multiple supporting points, counterarguments where appropriate, and a strong conclusion.'
    }
    
    instruction = length_instructions.get(length, length_instructions['medium'])
    system_prompt = f"""Argue AGAINST the topic. {instruction} Never identify as "con agent" - just argue directly."""
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

def create_con_agent(length: str = 'medium'):
    """Create the Con agent with Groq LLM"""
    return get_con_prompt(length) | get_con_llm()


@lru_cache(maxsize=1)
def get_judge_llm():
    """Get cached Judge LLM instance"""
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )

@lru_cache(maxsize=1)
def get_judge_prompt():
    """Get cached Judge prompt template"""
    system_prompt = """Judge this debate. Pick winner, list strengths (3 each), select best argument/rebuttal as exact quotes. Format:

Winner:
[Pro/Con]

Pro Strengths:
- [strength]

Con Strengths:
- [strength]

Best Argument:
"[quote]"

Best Rebuttal:
"[quote]"

Final Reasoning:
[reasoning]"""
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

def create_judge_agent():
    """Create the Judge agent with Groq LLM"""
    return get_judge_prompt() | get_judge_llm()


@lru_cache(maxsize=10)
def create_search_tool():
    """Create cached Tavily search tool for evidence retrieval"""
    return TavilySearchResults(
        max_results=2,  # Reduced from 3 for speed
        search_depth="basic",  # Changed from advanced for speed
        include_answer=True,
        include_raw_content=False,  # Disabled for speed
        include_images=False,
        api_key=TAVILY_API_KEY
    )


# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="DebateBot - Two Sides and a Judge",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# THEME CONFIGURATION
# ============================================================================

THEMES = {
    'midnight': {
        'name': '🌙 Midnight',
        'background': 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
        'sidebar': '#16213e',
        'card_bg': 'rgba(255, 255, 255, 0.05)',
        'card_border': 'rgba(255, 255, 255, 0.1)',
        'primary': '#6c5ce7',
        'secondary': '#a29bfe',
        'accent': '#00d4ff',
        'text': '#ffffff',
        'text_light': '#a29bfe',
        'success': '#4ecdc4',
        'danger': '#ff6b6b',
        'warning': '#ffd93d',
        'info': '#00d4ff',
        'gradient_1': '#6c5ce7',
        'gradient_2': '#a29bfe',
        'gradient_3': '#00d4ff',
        'button_bg': 'linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%)',
        'button_hover': 'linear-gradient(135deg, #a29bfe 0%, #00d4ff 100%)',
        'button_shadow': 'rgba(108, 92, 231, 0.3)',
        'pro_color': '#4ecdc4',
        'con_color': '#ff6b6b',
        'judge_color': '#00d4ff',
        'gold': '#ffd700',
        'scrollbar_track': '#0f3460',
        'scrollbar_thumb': 'linear-gradient(135deg, #6c5ce7, #a29bfe)',
    },
    'royal': {
        'name': '👑 Royal',
        'background': 'linear-gradient(135deg, #2c1e4a 0%, #4a2c7a 50%, #6b3fa0 100%)',
        'sidebar': '#4a2c7a',
        'card_bg': 'rgba(255, 255, 255, 0.08)',
        'card_border': 'rgba(156, 39, 176, 0.3)',
        'primary': '#9c27b0',
        'secondary': '#ba68c8',
        'accent': '#ffd700',
        'text': '#ffffff',
        'text_light': '#e1bee7',
        'success': '#4caf50',
        'danger': '#f44336',
        'warning': '#ff9800',
        'info': '#2196f3',
        'gradient_1': '#9c27b0',
        'gradient_2': '#ba68c8',
        'gradient_3': '#ffd700',
        'button_bg': 'linear-gradient(135deg, #9c27b0 0%, #ba68c8 100%)',
        'button_hover': 'linear-gradient(135deg, #ba68c8 0%, #ffd700 100%)',
        'button_shadow': 'rgba(156, 39, 176, 0.4)',
        'pro_color': '#9c27b0',
        'con_color': '#ff6b6b',
        'judge_color': '#ffd700',
        'gold': '#ffd700',
        'scrollbar_track': '#4a2c7a',
        'scrollbar_thumb': 'linear-gradient(135deg, #9c27b0, #ba68c8)',
    },
}


def get_active_theme():
    """Get the currently active theme"""
    theme_name = st.session_state.get('theme', 'royal')
    return THEMES.get(theme_name, THEMES['royal'])


class ThemeManager:
    """Centralized theme manager for handling all theme-related operations"""
    
    def __init__(self):
        self.themes = THEMES
        self.default_theme = 'royal'
    
    def get_theme(self, theme_name: str = None) -> dict:
        """Get theme configuration by name"""
        if theme_name is None:
            theme_name = st.session_state.get('theme', self.default_theme)
        return self.themes.get(theme_name, self.themes[self.default_theme])
    
    def get_active_theme(self) -> dict:
        """Get the currently active theme"""
        return self.get_theme()
    
    def set_theme(self, theme_name: str) -> None:
        """Set the active theme"""
        if theme_name in self.themes:
            st.session_state.theme = theme_name
            self.save_preference(theme_name)
    
    def save_preference(self, theme_name: str) -> None:
        """Save theme preference to file for persistence"""
        try:
            import json
            import os
            theme_file = os.path.join('config', 'theme.json')
            os.makedirs('config', exist_ok=True)
            with open(theme_file, 'w') as f:
                json.dump({'theme': theme_name}, f)
        except Exception as e:
            pass  # Silently fail if unable to save
    
    def load_preference(self) -> str:
        """Load theme preference from file"""
        try:
            import json
            import os
            theme_file = os.path.join('config', 'theme.json')
            if os.path.exists(theme_file):
                with open(theme_file, 'r') as f:
                    data = json.load(f)
                    return data.get('theme', self.default_theme)
        except Exception as e:
            pass
        return self.default_theme
    
    def get_all_themes(self) -> dict:
        """Get all available themes"""
        return self.themes
    
    def get_theme_names(self) -> list:
        """Get list of all theme names"""
        return list(self.themes.keys())
    
    def get_theme_display_name(self, theme_name: str) -> str:
        """Get display name for a theme"""
        theme = self.get_theme(theme_name)
        return theme.get('name', theme_name)


# Global theme manager instance
theme_manager = ThemeManager()


def save_theme_preference(theme_name: str):
    """Save theme preference to file for persistence (convenience function)"""
    theme_manager.save_preference(theme_name)


def load_theme_preference():
    """Load theme preference from file (convenience function)"""
    return theme_manager.load_preference()


def load_custom_css():
    """Load custom CSS with dynamic theme variables"""
    theme = get_active_theme()
    
    css = f"""
    <style>
    :root {{
        --theme-background: {theme['background']};
        --theme-sidebar: {theme['sidebar']};
        --theme-card-bg: {theme['card_bg']};
        --theme-card-border: {theme['card_border']};
        --theme-primary: {theme['primary']};
        --theme-secondary: {theme['secondary']};
        --theme-accent: {theme['accent']};
        --theme-text: {theme['text']};
        --theme-text-light: {theme['text_light']};
        --theme-success: {theme['success']};
        --theme-danger: {theme['danger']};
        --theme-warning: {theme['warning']};
        --theme-info: {theme['info']};
        --theme-gradient-1: {theme['gradient_1']};
        --theme-gradient-2: {theme['gradient_2']};
        --theme-gradient-3: {theme['gradient_3']};
        --theme-button-bg: {theme['button_bg']};
        --theme-button-hover: {theme['button_hover']};
        --theme-button-shadow: {theme['button_shadow']};
        --theme-pro-color: {theme['pro_color']};
        --theme-con-color: {theme['con_color']};
        --theme-judge-color: {theme['judge_color']};
        --theme-gold: {theme['gold']};
        --theme-scrollbar-track: {theme['scrollbar_track']};
        --theme-scrollbar-thumb: {theme['scrollbar_thumb']};
    }}
    
    .stApp {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: var(--theme-background);
    }}
    
    /* Animated background particles */
    @keyframes float {{
        0%, 100% {{ transform: translateY(0) rotate(0deg); }}
        50% {{ transform: translateY(-20px) rotate(180deg); }}
    }}
    
    .floating-particle {{
        animation: float 6s ease-in-out infinite;
    }}
    
    /* Lobby entrance animation */
    @keyframes slideInUp {{
        from {{ opacity: 0; transform: translateY(50px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .lobby-entrance {{
        animation: slideInUp 0.8s ease-out;
    }}
    
    /* Logo pulse animation */
    @keyframes logoPulse {{
        0%, 100% {{ transform: scale(1); filter: brightness(1); }}
        50% {{ transform: scale(1.05); filter: brightness(1.2); }}
    }}
    
    .logo-pulse {{
        animation: logoPulse 3s ease-in-out infinite;
    }}
    
    /* Gradient card */
    .gradient-card {{
        background: var(--theme-card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--theme-card-border);
        border-radius: 20px;
        padding: 30px;
        transition: all 0.3s ease;
    }}
    
    .gradient-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }}
    
    /* Game mode buttons */
    .game-mode-btn {{
        background: var(--theme-button-bg);
        border: none;
        border-radius: 15px;
        padding: 20px 40px;
        color: white;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 20px var(--theme-button-shadow);
    }}
    
    .game-mode-btn:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 30px var(--theme-button-shadow);
    }}
    
    .game-mode-btn:active {{
        transform: translateY(0) scale(0.98);
    }}
    
    /* Health bar animations */
    @keyframes healthPulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
    }}
    
    .health-pulse {{
        animation: healthPulse 2s infinite;
    }}
    
    /* Winner glow animation */
    @keyframes winnerGlow {{
        0%, 100% {{ box-shadow: 0 8px 16px rgba(0,0,0,0.3); }}
        50% {{ box-shadow: 0 8px 30px rgba(255, 215, 0, 0.5); }}
    }}
    
    .winner-glow {{
        animation: winnerGlow 2s infinite;
    }}
    
    /* Confetti animation */
    @keyframes confetti {{
        0% {{ transform: translateY(0) rotate(0deg); opacity: 1; }}
        100% {{ transform: translateY(100vh) rotate(720deg); opacity: 0; }}
    }}
    
    .confetti {{
        animation: confetti 3s ease-out forwards;
    }}
    
    /* Score animation */
    @keyframes scoreUp {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.3); color: var(--theme-gold); }}
        100% {{ transform: scale(1); }}
    }}
    
    .score-up {{
        animation: scoreUp 0.5s ease-out;
    }}
    
    /* Message slide animation */
    @keyframes messageSlide {{
        from {{ opacity: 0; transform: translateX(-20px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    .message-slide {{
        animation: messageSlide 0.3s ease-out;
    }}
    
    /* Typing indicator */
    @keyframes typing {{
        0%, 60%, 100% {{ transform: translateY(0); }}
        30% {{ transform: translateY(-5px); }}
    }}
    
    .typing-dot {{
        animation: typing 1.4s infinite ease-in-out;
    }}
    
    /* Gradient text */
    .gradient-text {{
        background: linear-gradient(135deg, var(--theme-gold), var(--theme-accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--theme-scrollbar-track);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--theme-scrollbar-thumb);
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, var(--theme-secondary), var(--theme-accent));
    }}
    
    /* Button hover effects */
    .stButton > button {{
        transition: all 0.3s ease;
        border-radius: 10px;
        background: var(--theme-button-bg);
        color: white;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px var(--theme-button-shadow);
        background: var(--theme-button-hover);
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: var(--theme-sidebar);
        transition: background 0.5s ease;
    }}
    
    [data-testid="stSidebar"] > div {{
        background: var(--theme-sidebar);
    }}
    
    /* Header styling */
    [data-testid="stHeader"] {{
        background: var(--theme-sidebar);
        transition: background 0.5s ease;
    }}
    
    /* Input boxes */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {{
        background: var(--theme-card-bg);
        color: var(--theme-text);
        border: 1px solid var(--theme-card-border);
        transition: all 0.3s ease;
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {{
        border-color: var(--theme-primary);
        box-shadow: 0 0 0 2px var(--theme-button-shadow);
    }}
    
    /* Dropdowns */
    .stSelectbox > div > div > select {{
        background: var(--theme-card-bg);
        color: var(--theme-text);
    }}
    
    /* Radio buttons */
    .stRadio > div {{
        color: var(--theme-text);
    }}
    
    .stRadio > label > div {{
        color: var(--theme-text);
    }}
    
    /* Sliders */
    .stSlider > div > div > div {{
        background: var(--theme-primary);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--theme-card-bg);
        border-bottom: 1px solid var(--theme-card-border);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: var(--theme-text);
    }}
    
    .stTabs [aria-selected="true"] {{
        color: var(--theme-primary);
        border-bottom: 2px solid var(--theme-primary);
    }}
    
    /* Dialogs/Modals */
    [data-testid="stModal"] {{
        background: var(--theme-card-bg);
        border: 2px solid var(--theme-card-border);
    }}
    
    /* Toast messages */
    .stToast {{
        background: var(--theme-card-bg);
        color: var(--theme-text);
        border: 1px solid var(--theme-card-border);
    }}
    
    /* Footer */
    footer {{
        visibility: hidden;
    }}
    
    /* Icons */
    .stIcon {{
        color: var(--theme-primary);
    }}
    
    /* Player cards */
    .player-card {{
        background: var(--theme-card-bg);
        border: 2px solid var(--theme-card-border);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        transition: all 0.3s ease;
    }}
    
    .player-card:hover {{
        border-color: var(--theme-gold);
        box-shadow: 0 10px 20px rgba(255, 215, 0, 0.2);
    }}
    
    /* Loading spinner */
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    .game-spinner {{
        border: 4px solid rgba(255,255,255,0.1);
        border-top: 4px solid var(--theme-primary);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }}
    
    /* Room code display */
    .room-code {{
        background: linear-gradient(135deg, var(--theme-success), var(--theme-info));
        padding: 15px 30px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        color: white;
        letter-spacing: 5px;
        text-align: center;
    }}
    
    /* Smooth theme transition - removed global transition to prevent page fade */
    
    /* Primary buttons */
    .stButton > button[kind="primary"] {{
        background: var(--theme-button-bg);
        color: white;
    }}
    
    .stButton > button[kind="primary"]:hover {{
        background: var(--theme-button-hover);
    }}
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {{
        background: var(--theme-secondary);
        color: white;
    }}
    
    .stButton > button[kind="secondary"]:hover {{
        background: var(--theme-accent);
    }}
    
    /* Info boxes */
    .stAlert {{
        background: var(--theme-card-bg);
        border: 1px solid var(--theme-card-border);
        color: var(--theme-text);
    }}
    
    /* Success messages */
    .stAlert[data-baseweb="toast"][data-type="success"] {{
        background: var(--theme-success);
        color: white;
    }}
    
    /* Error messages */
    .stAlert[data-baseweb="toast"][data-type="error"] {{
        background: var(--theme-danger);
        color: white;
    }}
    
    /* Warning messages */
    .stAlert[data-baseweb="toast"][data-type="warning"] {{
        background: var(--theme-warning);
        color: white;
    }}
    
    /* Info messages */
    .stAlert[data-baseweb="toast"][data-type="info"] {{
        background: var(--theme-info);
        color: white;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'lobby'
    
    if 'game_mode' not in st.session_state:
        st.session_state.game_mode = None
    
    if 'topic' not in st.session_state:
        st.session_state.topic = ''
    
    if 'num_rounds' not in st.session_state:
        st.session_state.num_rounds = 3
    
    if 'argument_length' not in st.session_state:
        st.session_state.argument_length = 'medium'
    
    if 'ai_difficulty' not in st.session_state:
        st.session_state.ai_difficulty = 'medium'
    
    if 'debate_style' not in st.session_state:
        st.session_state.debate_style = 'formal'
    
    if 'user_side' not in st.session_state:
        st.session_state.user_side = None
    
    if 'room_code' not in st.session_state:
        st.session_state.room_code = ''
    
    if 'is_host' not in st.session_state:
        st.session_state.is_host = False
    
    if 'max_players' not in st.session_state:
        st.session_state.max_players = 2
    
    if 'players' not in st.session_state:
        st.session_state.players = []
    
    if 'judge_personality' not in st.session_state:
        st.session_state.judge_personality = 'balanced'
    
    if 'theme' not in st.session_state:
        st.session_state.theme = theme_manager.load_preference()
    
    if 'show_citations' not in st.session_state:
        st.session_state.show_citations = True
    
    if 'debate_active' not in st.session_state:
        st.session_state.debate_active = False
    
    if 'debate_complete' not in st.session_state:
        st.session_state.debate_complete = False
    
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 0
    
    if 'pro_score' not in st.session_state:
        st.session_state.pro_score = 50.0
    
    if 'con_score' not in st.session_state:
        st.session_state.con_score = 50.0
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'citations' not in st.session_state:
        st.session_state.citations = []
    
    if 'judge_verdict' not in st.session_state:
        st.session_state.judge_verdict = None
    
    if 'show_transcript' not in st.session_state:
        st.session_state.show_transcript = False
    
    if 'debate_history' not in st.session_state:
        st.session_state.debate_history = []
    
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ''
    
    if 'current_turn' not in st.session_state:
        st.session_state.current_turn = 'pro'
    
    if 'pro_spoken_this_round' not in st.session_state:
        st.session_state.pro_spoken_this_round = False
    
    if 'con_spoken_this_round' not in st.session_state:
        st.session_state.con_spoken_this_round = False
    
    if 'show_round_transcript' not in st.session_state:
        st.session_state.show_round_transcript = False
    
    if 'show_leave_confirmation' not in st.session_state:
        st.session_state.show_leave_confirmation = False
    
    if 'automating' not in st.session_state:
        st.session_state.automating = False


def reset_debate_state():
    """Reset debate state for new debate"""
    st.session_state.debate_active = False
    st.session_state.debate_complete = False
    st.session_state.current_round = 1
    st.session_state.pro_score = 50.0
    st.session_state.con_score = 50.0
    st.session_state.messages = []
    st.session_state.citations = []
    st.session_state.judge_verdict = None
    st.session_state.show_transcript = False
    st.session_state.current_turn = 'pro'
    st.session_state.pro_spoken_this_round = False
    st.session_state.con_spoken_this_round = False
    st.session_state.show_round_transcript = False
    st.session_state.show_leave_confirmation = False


# ============================================================================
# HISTORY MANAGEMENT
# ============================================================================

def save_debate_to_history():
    """Save current debate to history"""
    debate_data = {
        'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'topic': st.session_state.topic,
        'timestamp': datetime.now().isoformat(),
        'num_rounds': st.session_state.num_rounds,
        'messages': [msg.content for msg in st.session_state.messages],
        'citations': st.session_state.citations,
        'verdict': st.session_state.judge_verdict,
        'pro_score': st.session_state.pro_score,
        'con_score': st.session_state.con_score
    }
    
    history_file = os.path.join('history', 'debates.json')
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(debate_data)
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    st.session_state.debate_history = history


def load_debate_history():
    """Load debate history from file"""
    history_file = os.path.join('history', 'debates.json')
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []


def delete_debate(debate_id):
    """Delete a debate from history"""
    history_file = os.path.join('history', 'debates.json')
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        history = [d for d in history if d['id'] != debate_id]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        st.session_state.debate_history = history


# ============================================================================
# PDF EXPORT
# ============================================================================

def export_debate_to_pdf():
    """Export debate to PDF"""
    filename = f"exports/debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = styles['Title']
    title_style.alignment = 1
    story.append(Paragraph("DebateBot - Debate Transcript", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Topic
    story.append(Paragraph(f"<b>Topic:</b> {st.session_state.topic}", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))
    
    # Date
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    
    # Messages
    story.append(Paragraph("<b>Debate Transcript:</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))
    
    for i, msg in enumerate(st.session_state.messages):
        role = msg.name if hasattr(msg, 'name') and msg.name else ("PRO AGENT" if i % 2 == 0 else "CON AGENT")
        color = "#00ff00" if role == "PRO AGENT" or role == "Pro" else "#ff0000"
        story.append(Paragraph(f"<b>{role.upper()}:</b>", styles['Normal']))
        story.append(Paragraph(msg.content, styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))
    
    # Citations
    if st.session_state.citations:
        story.append(Paragraph("<b>Citations:</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        
        for citation in st.session_state.citations:
            story.append(Paragraph(f"<b>Source:</b> {citation.get('source', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"<b>URL:</b> {citation.get('url', 'N/A')}", styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))
    
    # Verdict
    if st.session_state.judge_verdict:
        story.append(Paragraph("<b>Judge's Verdict:</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        
        verdict = st.session_state.judge_verdict
        story.append(Paragraph(f"<b>Winner:</b> {verdict.get('winner', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Final Score:</b> {verdict.get('final_score', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Best Argument:</b> {verdict.get('best_argument', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Best Rebuttal:</b> {verdict.get('best_rebuttal', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Pro Strengths:</b> {verdict.get('pro_strengths', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Con Strengths:</b> {verdict.get('con_strengths', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Final Reasoning:</b> {verdict.get('final_reasoning', 'N/A')}", styles['Normal']))
    
    doc.build(story)
    return filename


# ============================================================================
# DEBATE LOGIC
# ============================================================================

def get_pro_argument(topic: str, round_num: int) -> str:
    """Get Pro agent's argument using Groq"""
    try:
        pro_agent = create_pro_agent()
        
        # Only send essential context: topic, round, opponent's last argument
        messages = [SystemMessage(content=f"Topic: {topic}. Round {round_num + 1}. Argue FOR.")]
        
        # Add only opponent's last argument for context
        if st.session_state.messages:
            last_msg = st.session_state.messages[-1]
            if last_msg.name == "Con":
                messages.append(HumanMessage(content=f"Opponent said: {last_msg.content}"))
        
        response = pro_agent.invoke({"messages": messages})
        return response.content
    except Exception as e:
        return f"Error generating Pro argument: {str(e)}"


def get_con_rebuttal(topic: str, round_num: int) -> str:
    """Get Con agent's rebuttal using Groq"""
    try:
        con_agent = create_con_agent()
        
        # Only send essential context: topic, round, opponent's last argument
        messages = [SystemMessage(content=f"Topic: {topic}. Round {round_num + 1}. Argue AGAINST.")]
        
        # Add only opponent's last argument for context
        if st.session_state.messages:
            last_msg = st.session_state.messages[-1]
            if last_msg.name == "Pro":
                messages.append(HumanMessage(content=f"Opponent said: {last_msg.content}"))
        
        response = con_agent.invoke({"messages": messages})
        return response.content
    except Exception as e:
        return f"Error generating Con rebuttal: {str(e)}"


def get_judge_verdict() -> Dict[str, str]:
    """Get judge's verdict using Groq"""
    try:
        judge_agent = create_judge_agent()
        
        game_mode = st.session_state.get('game_mode', 'ai_vs_ai')
        user_side = st.session_state.get('user_side', 'pro')
        
        # Optimized context - only essential information
        debate_context = f"""Topic: {st.session_state.topic}
Rounds: {st.session_state.num_rounds}
Scores: Pro {int(st.session_state.pro_score)}% - Con {int(st.session_state.con_score)}%

IMPORTANT INSTRUCTIONS:
- Evaluate ONLY the arguments that actually appear in the transcript below.
- DO NOT invent, fabricate, or hallucinate strengths, evidence, examples, or reasoning that are not present in the actual arguments.
- If an argument is weak, short, meaningless, or lacks evidence, honestly state that it is weak/lacks evidence.
- DO NOT make weak arguments appear stronger than they are.
- Only identify strengths that are clearly present in the actual text.

Transcript:
"""
        
        # Add debate messages (simplified format) - use actual message names
        for i, msg in enumerate(st.session_state.messages):
            speaker = msg.name if hasattr(msg, 'name') and msg.name else ("Pro" if i % 2 == 0 else "Con")
            debate_context += f"{speaker}: {msg.content}\n"
        
        messages = [SystemMessage(content=debate_context)]
        messages.append(HumanMessage(content="""Analyze the debate transcript above and provide a verdict. You MUST use this EXACT format with these EXACT headers:

Winner: [Pro or Con or Draw]

Pro Strengths:
- [specific strength from Pro's actual arguments]
- [another specific strength]
...

Con Strengths:
- [specific strength from Con's actual arguments]
- [another specific strength]
...

Best Argument: [the strongest actual argument from the transcript, quoted or described]

Best Rebuttal: [the strongest actual rebuttal from the transcript, quoted or described]

Final Reasoning: [your detailed explanation]

IMPORTANT: Only identify strengths, arguments, and rebuttals that ACTUALLY appear in the transcript. Do not invent or fabricate anything. If a side has no clear strengths, state that honestly. If there is no clear best argument or rebuttal, state that honestly."""))
        
        response = judge_agent.invoke({"messages": messages})
        verdict_text = response.content
        
        # DEBUG: Print raw Judge response
        print("=" * 50)
        print("DEBUG: Raw Judge Response:")
        print("=" * 50)
        print(verdict_text)
        print("=" * 50)
        
        # DEBUG: Print transcript
        print("DEBUG: Debate Transcript:")
        print("=" * 50)
        print(debate_context)
        print("=" * 50)
        
        # Parse verdict with improved parsing
        verdict = {
            'winner': 'Draw',
            'final_score': f"Pro: {int(st.session_state.pro_score)} - Con: {int(st.session_state.con_score)}",
            'best_argument': 'Not specified',
            'best_rebuttal': 'Not specified',
            'pro_strengths': 'Not specified',
            'con_strengths': 'Not specified',
            'final_reasoning': verdict_text
        }
        
        # Parse Winner (case-insensitive)
        lines = verdict_text.split('\n')
        for i, line in enumerate(lines):
            if 'winner:' in line.lower():
                winner_text = line.split(':', 1)[1].strip() if ':' in line else ''
                # Get next lines if winner spans multiple lines
                j = i + 1
                while j < len(lines) and lines[j].strip() and not any(marker.lower() in lines[j].lower() for marker in ['pro strengths', 'con strengths', 'best argument', 'best rebuttal', 'final reasoning']):
                    winner_text += ' ' + lines[j].strip()
                    j += 1
                if winner_text:
                    verdict['winner'] = winner_text
                break
        
        # Parse Pro Strengths (case-insensitive)
        for i, line in enumerate(lines):
            if 'pro strengths' in line.lower():
                strengths = []
                j = i + 1
                while j < len(lines) and (lines[j].strip().startswith('-') or lines[j].strip().startswith('*') or lines[j].strip().startswith('•')):
                    strengths.append(lines[j].strip().lstrip('-*•').strip())
                    j += 1
                if strengths:
                    verdict['pro_strengths'] = '\n'.join([f"- {s}" for s in strengths])
                break
        
        # Parse Con Strengths (case-insensitive)
        for i, line in enumerate(lines):
            if 'con strengths' in line.lower():
                strengths = []
                j = i + 1
                while j < len(lines) and (lines[j].strip().startswith('-') or lines[j].strip().startswith('*') or lines[j].strip().startswith('•')):
                    strengths.append(lines[j].strip().lstrip('-*•').strip())
                    j += 1
                if strengths:
                    verdict['con_strengths'] = '\n'.join([f"- {s}" for s in strengths])
                break
        
        # Parse Best Argument (case-insensitive)
        for i, line in enumerate(lines):
            if 'best argument' in line.lower():
                arg_text = line.split(':', 1)[1].strip() if ':' in line else ''
                # Remove quotes if present
                arg_text = arg_text.strip('"\'')
                # Get next lines if argument spans multiple lines
                j = i + 1
                while j < len(lines) and lines[j].strip() and not any(marker.lower() in lines[j].lower() for marker in ['best rebuttal', 'final reasoning']):
                    arg_text += ' ' + lines[j].strip()
                    j += 1
                if arg_text:
                    verdict['best_argument'] = arg_text.strip('"\'')
                break
        
        # Parse Best Rebuttal (case-insensitive)
        for i, line in enumerate(lines):
            if 'best rebuttal' in line.lower():
                rebuttal_text = line.split(':', 1)[1].strip() if ':' in line else ''
                # Remove quotes if present
                rebuttal_text = rebuttal_text.strip('"\'')
                # Get next lines if rebuttal spans multiple lines
                j = i + 1
                while j < len(lines) and lines[j].strip() and 'final reasoning' not in lines[j].lower():
                    rebuttal_text += ' ' + lines[j].strip()
                    j += 1
                if rebuttal_text:
                    verdict['best_rebuttal'] = rebuttal_text.strip('"\'')
                break
        
        # Parse Final Reasoning (case-insensitive)
        for i, line in enumerate(lines):
            if 'final reasoning' in line.lower():
                reasoning = line.split(':', 1)[1].strip() if ':' in line else ''
                j = i + 1
                while j < len(lines):
                    reasoning += '\n' + lines[j]
                    j += 1
                if reasoning:
                    verdict['final_reasoning'] = reasoning.strip()
                break
        
        # For Human vs AI mode, convert Pro/Con to Human/AI labels (AFTER parsing)
        if game_mode == 'human_vs_ai':
            # Map Pro/Con to Human/AI based on user_side
            if user_side == 'pro':
                # Human is Pro, AI is Con
                verdict['winner'] = verdict['winner'].replace('Pro', 'Human').replace('Con', 'AI')
                verdict['final_score'] = verdict['final_score'].replace('Pro', 'Human').replace('Con', 'AI')
                verdict['pro_strengths'] = verdict['pro_strengths'].replace('Pro', 'Human')
                verdict['con_strengths'] = verdict['con_strengths'].replace('Con', 'AI')
            else:
                # Human is Con, AI is Pro
                verdict['winner'] = verdict['winner'].replace('Con', 'Human').replace('Pro', 'AI')
                verdict['final_score'] = verdict['final_score'].replace('Con', 'Human').replace('Pro', 'AI')
                verdict['pro_strengths'] = verdict['pro_strengths'].replace('Pro', 'AI')
                verdict['con_strengths'] = verdict['con_strengths'].replace('Con', 'Human')
        
        # DEBUG: Print parsed verdict values
        print("DEBUG: Parsed Verdict Values:")
        print("=" * 50)
        print(f"pro_strengths: {verdict['pro_strengths']}")
        print(f"con_strengths: {verdict['con_strengths']}")
        print(f"best_argument: {verdict['best_argument']}")
        print(f"best_rebuttal: {verdict['best_rebuttal']}")
        print(f"winner: {verdict['winner']}")
        print("=" * 50)
        
        # Store AI vs AI Judge result in session state
        if game_mode == 'ai_vs_ai':
            st.session_state["ai_vs_ai_judge_result"] = verdict.copy()
            print("DEBUG: Stored AI vs AI Judge result in session_state")
        
        return verdict
    except Exception as e:
        return {
            'winner': 'Draw',
            'final_score': f"Pro: {int(st.session_state.pro_score)} - Con: {int(st.session_state.con_score)}",
            'best_argument': 'Error in judgment',
            'best_rebuttal': 'Error in judgment',
            'pro_strengths': 'Error in judgment',
            'con_strengths': 'Error in judgment',
            'final_reasoning': f"Error generating verdict: {str(e)}"
        }


def search_tavily(query: str) -> List[Dict[str, Any]]:
    """Search Tavily for evidence"""
    try:
        search_tool = create_search_tool()
        results = search_tool.invoke({"query": query})
        
        citations = []
        for result in results:
            if isinstance(result, dict):
                citations.append({
                    'source': result.get('title', 'Unknown'),
                    'url': result.get('url', ''),
                    'content': result.get('content', '')
                })
        
        return citations
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []


def simulate_pro_turn():
    """Simulate Pro agent's turn with streaming"""
    topic = st.session_state.topic
    round_num = st.session_state.current_round
    argument_length = st.session_state.get('argument_length', 'medium')
    
    # Pro Agent's turn - immediate loading indicator
    placeholder = st.empty()
    placeholder.markdown("""
    <div style="text-align: center; padding: 20px;">
        <div style="color: var(--theme-pro-color); font-size: 24px; font-weight: bold;">
            🟢 Pro Agent is thinking...
        </div>
        <div style="display: flex; justify-content: center; gap: 8px; margin-top: 10px;">
            <div class="typing-dot" style="width: 10px; height: 10px; background: var(--theme-pro-color); border-radius: 50%;"></div>
            <div class="typing-dot" style="width: 10px; height: 10px; background: var(--theme-pro-color); border-radius: 50%;"></div>
            <div class="typing-dot" style="width: 10px; height: 10px; background: var(--theme-pro-color); border-radius: 50%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get argument with streaming using dynamic length
    pro_agent = create_pro_agent(argument_length)
    messages = [SystemMessage(content=f"Topic: {topic}. Round {round_num + 1}. Argue FOR.")]
    
    if st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg.name == "Con":
            messages.append(HumanMessage(content=f"Opponent said: {last_msg.content}"))
    
    # Stream the response
    full_response = ""
    try:
        for chunk in pro_agent.stream({"messages": messages}):
            if hasattr(chunk, 'content'):
                full_response += chunk.content
    except:
        # Fallback to invoke if streaming fails
        response = pro_agent.invoke({"messages": messages})
        full_response = response.content
    
    placeholder.empty()
    
    st.session_state.messages.append(AIMessage(content=full_response, name="Pro"))
    st.session_state.pro_spoken_this_round = True
    st.session_state.current_turn = 'con'
    
    # Update scores
    pro_change = random.uniform(-5, 10)
    con_change = random.uniform(-10, 5)
    st.session_state.pro_score = max(0, min(100, st.session_state.pro_score + pro_change))
    st.session_state.con_score = max(0, min(100, st.session_state.con_score + con_change))


def simulate_con_turn():
    """Simulate Con agent's turn with streaming"""
    topic = st.session_state.topic
    round_num = st.session_state.current_round
    argument_length = st.session_state.get('argument_length', 'medium')
    
    # Con Agent's turn - immediate loading indicator
    placeholder = st.empty()
    placeholder.markdown("""
    <div style="text-align: center; padding: 20px;">
        <div style="color: var(--theme-con-color); font-size: 24px; font-weight: bold;">
            🔴 Con Agent is thinking...
        </div>
        <div style="display: flex; justify-content: center; gap: 8px; margin-top: 10px;">
            <div class="typing-dot" style="width: 10px; height: 10px; background: var(--theme-con-color); border-radius: 50%;"></div>
            <div class="typing-dot" style="width: 10px; height: 10px; background: var(--theme-con-color); border-radius: 50%;"></div>
            <div class="typing-dot" style="width: 10px; height: 10px; background: var(--theme-con-color); border-radius: 50%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get rebuttal with streaming using dynamic length
    con_agent = create_con_agent(argument_length)
    messages = [SystemMessage(content=f"Topic: {topic}. Round {round_num + 1}. Argue AGAINST.")]
    
    if st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg.name == "Pro":
            messages.append(HumanMessage(content=f"Opponent said: {last_msg.content}"))
    
    # Stream the response
    full_response = ""
    try:
        for chunk in con_agent.stream({"messages": messages}):
            if hasattr(chunk, 'content'):
                full_response += chunk.content
    except:
        # Fallback to invoke if streaming fails
        response = con_agent.invoke({"messages": messages})
        full_response = response.content
    
    placeholder.empty()
    
    st.session_state.messages.append(AIMessage(content=full_response, name="Con"))
    st.session_state.con_spoken_this_round = True
    st.session_state.current_turn = 'pro'
    
    # Update scores
    pro_change = random.uniform(-10, 5)
    con_change = random.uniform(-5, 10)
    st.session_state.pro_score = max(0, min(100, st.session_state.pro_score + pro_change))
    st.session_state.con_score = max(0, min(100, st.session_state.con_score + con_change))
    
    # Fetch citations for the round after both agents have spoken
    if st.session_state.show_citations:
        citations = search_tavily(f"{topic} debate evidence")
        st.session_state.citations.extend(citations)


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_health_bars():
    """Render health/persuasion bars"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 10px;">
            <div style="font-weight: bold; color: var(--theme-pro-color); font-size: 18px;">PRO AGENT</div>
            <div style="background: var(--theme-card-bg); border: 2px solid var(--theme-pro-color); border-radius: 10px; height: 30px; margin: 10px 0; position: relative; overflow: hidden;">
                <div style="background: linear-gradient(90deg, var(--theme-pro-color), var(--theme-success)); height: 100%; width: {st.session_state.pro_score}%; transition: width 0.5s ease;"></div>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-weight: bold; text-shadow: 1px 1px 2px black;">{int(st.session_state.pro_score)}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; margin: 15px 0;">
            <div style="background: var(--theme-gold); color: black; padding: 10px 20px; border-radius: 20px; font-weight: bold; font-size: 20px; display: inline-block; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                ROUND {st.session_state.current_round} / {st.session_state.num_rounds}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 10px;">
            <div style="font-weight: bold; color: var(--theme-con-color); font-size: 18px;">CON AGENT</div>
            <div style="background: var(--theme-card-bg); border: 2px solid var(--theme-con-color); border-radius: 10px; height: 30px; margin: 10px 0; position: relative; overflow: hidden;">
                <div style="background: linear-gradient(90deg, var(--theme-danger), var(--theme-con-color)); height: 100%; width: {st.session_state.con_score}%; transition: width 0.5s ease;"></div>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-weight: bold; text-shadow: 1px 1px 2px black;">{int(st.session_state.con_score)}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_progress_indicator():
    """Render debate progress indicator"""
    progress = ((st.session_state.current_round - 1) / st.session_state.num_rounds) * 100
    st.markdown(f"""
    <div style="margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: var(--theme-text); font-weight: bold;">Debate Progress</span>
            <span style="color: var(--theme-gold); font-weight: bold;">{int(progress)}%</span>
        </div>
        <div style="background: var(--theme-card-bg); border: 2px solid var(--theme-card-border); border-radius: 10px; height: 12px; position: relative; overflow: hidden;">
            <div style="background: linear-gradient(90deg, var(--theme-gold), var(--theme-accent)); height: 100%; width: {progress}%; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_message(message: AIMessage, index: int):
    """Render a single debate message"""
    is_pro = index % 2 == 0
    bg_color = 'var(--theme-card-bg)' if is_pro else 'var(--theme-card-bg)'
    border_color = 'var(--theme-pro-color)' if is_pro else 'var(--theme-con-color)'
    align = 'left' if is_pro else 'right'
    agent_name = 'PRO AGENT' if is_pro else 'CON AGENT'
    
    st.markdown(f"""
    <div style="margin: 15px 0;">
        <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 15px; padding: 15px 20px; max-width: 80%; margin: {'0 auto 0 0' if align == 'left' else '0 0 0 auto'}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: {border_color}; font-weight: bold; font-size: 14px; margin-bottom: 10px;">{agent_name}</div>
            <div style="color: var(--theme-text); line-height: 1.6;">{message.content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_scoreboard():
    """Render scoreboard"""
    leader = "PRO" if st.session_state.pro_score > st.session_state.con_score else ("CON" if st.session_state.con_score > st.session_state.pro_score else "TIE")
    leader_color = "var(--theme-pro-color)" if leader == "PRO" else "var(--theme-con-color)" if leader == "CON" else "var(--theme-gold)"
    
    st.markdown(f"""
    <div style="background: var(--theme-card-bg); border: 2px solid var(--theme-gold); border-radius: 15px; padding: 20px; margin: 15px 0;">
        <div style="color: var(--theme-gold); font-weight: bold; font-size: 16px; margin-bottom: 15px; text-align: center;">📊 SCOREBOARD</div>
        <div style="display: flex; justify-content: space-around; align-items: center;">
            <div style="text-align: center;">
                <div style="color: var(--theme-pro-color); font-weight: bold; font-size: 24px;">{int(st.session_state.pro_score)}%</div>
                <div style="color: var(--theme-text); font-size: 14px;">PRO</div>
            </div>
            <div style="text-align: center;">
                <div style="color: {leader_color}; font-weight: bold; font-size: 18px;">{leader}</div>
                <div style="color: var(--theme-gold); font-size: 14px;">LEADER</div>
            </div>
            <div style="text-align: center;">
                <div style="color: var(--theme-con-color); font-weight: bold; font-size: 24px;">{int(st.session_state.con_score)}%</div>
                <div style="color: var(--theme-text); font-size: 14px;">CON</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGES
# ============================================================================

def render_lobby():
    """Render game-style lobby with three main modes"""
    st.markdown("""
    <div class="lobby-entrance" style="text-align: center; padding: 40px 20px;">
        <div class="logo-pulse" style="margin-bottom: 20px;">
            <h1 style="background: linear-gradient(135deg, var(--theme-gold), var(--theme-accent), var(--theme-secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 64px; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                ⚔️ DebateBot
            </h1>
        </div>
        <h2 style="color: var(--theme-text); font-size: 28px; margin-bottom: 10px; font-weight: normal;">
            The Ultimate Debate Arena
        </h2>
        <p style="color: var(--theme-text-light); font-size: 18px; max-width: 700px; margin: 0 auto 40px auto; line-height: 1.6;">
            Battle it out with AI or challenge your friends in intense intellectual combat!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Game mode cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="gradient-card" style="text-align: center; margin: 20px 0;">
            <div style="font-size: 48px; margin-bottom: 15px;">🤖</div>
            <h3 style="color: var(--theme-success); font-size: 24px; margin-bottom: 10px;">AI vs AI</h3>
            <p style="color: var(--theme-text-light); font-size: 14px; margin-bottom: 20px;">Watch two AI agents debate each other</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎮 Play", key="ai_vs_ai", use_container_width=True, type="primary"):
            st.session_state.game_mode = 'ai_vs_ai'
            st.session_state.current_page = 'ai_vs_ai_config'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="gradient-card" style="text-align: center; margin: 20px 0;">
            <div style="font-size: 48px; margin-bottom: 15px;">👤</div>
            <h3 style="color: var(--theme-danger); font-size: 24px; margin-bottom: 10px;">Human vs AI</h3>
            <p style="color: var(--theme-text-light); font-size: 14px; margin-bottom: 20px;">Challenge an AI to a debate</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎮 Play", key="human_vs_ai", use_container_width=True, type="primary"):
            st.session_state.game_mode = 'human_vs_ai'
            st.session_state.current_page = 'human_vs_ai_config'
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="gradient-card" style="text-align: center; margin: 20px 0;">
            <div style="font-size: 48px; margin-bottom: 15px;">👥</div>
            <h3 style="color: var(--theme-info); font-size: 24px; margin-bottom: 10px;">Play with Friends</h3>
            <p style="color: var(--theme-text-light); font-size: 14px; margin-bottom: 20px;">Create or join a multiplayer room</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎮 Play", key="multiplayer", use_container_width=True, type="primary"):
            st.session_state.game_mode = 'multiplayer'
            st.session_state.current_page = 'multiplayer_lobby'
            st.rerun()
    
    st.markdown("---")
    
    # Bottom buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("⚙️ Settings", key="lobby_settings", use_container_width=True):
            st.session_state.current_page = 'settings'
            st.rerun()
    with col2:
        if st.button("🎨 Themes", key="lobby_themes", use_container_width=True):
            st.session_state.current_page = 'themes'
            st.rerun()
    with col3:
        if st.button("❓ Help", key="lobby_help", use_container_width=True):
            st.session_state.current_page = 'help'
            st.rerun()
    with col4:
        if st.button("📜 History", key="lobby_history", use_container_width=True):
            st.session_state.current_page = 'history'
            st.rerun()


def render_ai_vs_ai_config():
    """Render AI vs AI configuration screen"""
    st.markdown("""
    <div style="text-align: center; padding: 30px 20px;">
        <h2 style="color: var(--theme-success); font-size: 36px; margin-bottom: 10px;">🤖 AI vs AI</h2>
        <p style="color: var(--theme-text-light); font-size: 16px;">Configure your AI debate</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Configuration form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🎯 Debate Topic</div>", unsafe_allow_html=True)
        topic = st.text_input("Enter topic:", placeholder="Should AI be regulated?", key="ai_vs_ai_topic")
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🔄 Number of Rounds</div>", unsafe_allow_html=True)
        num_rounds = st.selectbox("Rounds:", [3, 5, 7], index=0, key="ai_vs_ai_rounds")
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>📏 Argument Length</div>", unsafe_allow_html=True)
        arg_length = st.selectbox("Length:", ["short", "medium", "long"], index=1, key="ai_vs_ai_length")
    
    with col2:
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🎮 AI Difficulty</div>", unsafe_allow_html=True)
        difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard"], index=1, key="ai_vs_ai_difficulty")
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🎭 Debate Style</div>", unsafe_allow_html=True)
        style = st.selectbox("Style:", ["formal", "casual", "competitive"], index=0, key="ai_vs_ai_style")
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>⚖️ Judge Personality</div>", unsafe_allow_html=True)
        judge = st.selectbox("Judge:", ["balanced", "strict", "lenient"], index=0, key="ai_vs_ai_judge")
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Back", key="ai_vs_ai_back", use_container_width=True):
            st.session_state.current_page = 'lobby'
            st.rerun()
    with col2:
        if st.button("⚔️ Start Debate", key="ai_vs_ai_start", use_container_width=True, type="primary"):
            if topic and topic.strip():
                st.session_state.topic = topic.strip()
                st.session_state.num_rounds = num_rounds
                st.session_state.argument_length = arg_length
                st.session_state.ai_difficulty = difficulty
                st.session_state.debate_style = style
                st.session_state.judge_personality = judge
                st.session_state.game_mode = 'ai_vs_ai'
                st.session_state.current_page = 'debate'
                reset_debate_state()
                st.session_state.debate_active = True
                st.rerun()
            else:
                st.error("Please enter a debate topic.")


def render_human_vs_ai_config():
    """Render Human vs AI configuration dialog"""
    st.markdown("""
    <div style="text-align: center; padding: 30px 20px;">
        <h2 style="color: var(--theme-danger); font-size: 36px; margin-bottom: 10px;">👤 Human vs AI</h2>
        <p style="color: var(--theme-text-light); font-size: 16px;">Configure your debate</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Configuration form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🎯 Debate Topic</div>", unsafe_allow_html=True)
        topic = st.text_input("Enter topic:", placeholder="Should AI be regulated?", key="human_vs_ai_topic")
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🔄 Number of Rounds</div>", unsafe_allow_html=True)
        num_rounds = st.selectbox("Rounds:", [3, 5, 7], index=0, key="human_vs_ai_rounds")
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>📏 Argument Length</div>", unsafe_allow_html=True)
        arg_length = st.selectbox("Length:", ["short", "medium", "long"], index=1, key="human_vs_ai_length")
    
    with col2:
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🎮 AI Difficulty</div>", unsafe_allow_html=True)
        difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard"], index=1, key="human_vs_ai_difficulty")
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🎭 Debate Style</div>", unsafe_allow_html=True)
        style = st.selectbox("Style:", ["formal", "casual", "competitive"], index=0, key="human_vs_ai_style")
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>⚔️ Choose Your Side</div>", unsafe_allow_html=True)
        side = st.radio("Side:", ["In Favour", "Against"], key="human_vs_ai_side")
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Back", key="human_vs_ai_back", use_container_width=True):
            st.session_state.current_page = 'lobby'
            st.rerun()
    with col2:
        if st.button("⚔️ Start Debate", key="human_vs_ai_start", use_container_width=True, type="primary"):
            if topic and topic.strip():
                st.session_state.topic = topic.strip()
                st.session_state.num_rounds = num_rounds
                st.session_state.argument_length = arg_length
                st.session_state.ai_difficulty = difficulty
                st.session_state.debate_style = style
                st.session_state.user_side = 'pro' if side == "In Favour" else 'con'
                st.session_state.game_mode = 'human_vs_ai'
                st.session_state.current_page = 'debate'
                reset_debate_state()
                st.session_state.debate_active = True
                st.rerun()
            else:
                st.error("Please enter a debate topic.")


def render_multiplayer_lobby():
    """Render multiplayer lobby with create/join room"""
    st.markdown("""
    <div style="text-align: center; padding: 30px 20px;">
        <h2 style="color: var(--theme-info); font-size: 36px; margin-bottom: 10px;">👥 Play with Friends</h2>
        <p style="color: var(--theme-text-light); font-size: 16px;">Create or join a debate room</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create or Join tabs
    tab1, tab2 = st.tabs(["🏠 Create Room", "🔑 Join Room"])
    
    with tab1:
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🎯 Debate Topic</div>", unsafe_allow_html=True)
        topic = st.text_input("Enter topic:", placeholder="Should AI be regulated?", key="create_room_topic")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>👥 Max Players</div>", unsafe_allow_html=True)
            max_players = st.selectbox("Players:", [2, 3, 4, 5, 6], index=0, key="create_room_players")
        with col2:
            st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🔄 Rounds</div>", unsafe_allow_html=True)
            num_rounds = st.selectbox("Rounds:", [3, 5, 7], index=0, key="create_room_rounds")
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>📏 Argument Length</div>", unsafe_allow_html=True)
        arg_length = st.selectbox("Length:", ["short", "medium", "long"], index=1, key="create_room_length")
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🔒 Room Type</div>", unsafe_allow_html=True)
        room_type = st.radio("Type:", ["Public", "Private"], key="create_room_type")
        
        if st.button("🏠 Create Room", key="create_room_btn", use_container_width=True, type="primary"):
            if topic and topic.strip():
                # Generate room code
                import random
                import string
                room_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                st.session_state.room_code = room_code
                st.session_state.topic = topic.strip()
                st.session_state.max_players = max_players
                st.session_state.num_rounds = num_rounds
                st.session_state.argument_length = arg_length
                st.session_state.is_host = True
                st.session_state.players = ["Host"]
                st.session_state.current_page = 'room_lobby'
                st.rerun()
            else:
                st.error("Please enter a debate topic.")
    
    with tab2:
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🔑 Room Code</div>", unsafe_allow_html=True)
        room_code = st.text_input("Enter code:", placeholder="A4JX8Q", key="join_room_code").upper()
        
        player_name = st.text_input("Your Name:", placeholder="Player", key="join_room_name")
        
        if st.button("🔑 Join Room", key="join_room_btn", use_container_width=True, type="primary"):
            if room_code and player_name:
                st.session_state.room_code = room_code
                st.session_state.is_host = False
                st.session_state.players.append(player_name)
                st.session_state.current_page = 'room_lobby'
                st.rerun()
            else:
                st.error("Please enter room code and your name.")
    
    st.markdown("---")
    
    if st.button("⬅️ Back to Lobby", key="multiplayer_back", use_container_width=True):
        st.session_state.current_page = 'lobby'
        st.rerun()


def render_room_lobby():
    """Render room lobby waiting for players"""
    st.markdown(f"""
    <div style="text-align: center; padding: 30px 20px;">
        <h2 style="color: var(--theme-info); font-size: 36px; margin-bottom: 10px;">🏠 Room Lobby</h2>
        <div class="room-code">{st.session_state.room_code}</div>
        <p style="color: var(--theme-text-light); font-size: 16px; margin-top: 20px;">Topic: {st.session_state.topic}</p>
        <p style="color: var(--theme-text-light); font-size: 14px;">Players: {len(st.session_state.players)} / {st.session_state.max_players}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Player cards
    for i, player in enumerate(st.session_state.players):
        st.markdown(f"""
        <div class="player-card">
            <div style="color: var(--theme-success); font-weight: bold; font-size: 18px;">{player}</div>
            <div style="color: var(--theme-text-light); font-size: 14px;">Ready</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Waiting slots
    for i in range(st.session_state.max_players - len(st.session_state.players)):
        st.markdown("""
        <div class="player-card" style="opacity: 0.5;">
            <div style="color: var(--theme-text-light); font-weight: bold; font-size: 18px;">Waiting...</div>
            <div style="color: var(--theme-text-light); font-size: 14px;">Empty slot</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Host controls
    if st.session_state.is_host:
        if st.button("⚔️ Start Debate", key="start_multiplayer", use_container_width=True, type="primary"):
            st.session_state.game_mode = 'multiplayer'
            st.session_state.current_page = 'debate'
            reset_debate_state()
            st.session_state.debate_active = True
            st.rerun()
    else:
        st.info("Waiting for host to start the debate...")
    
    if st.button("⬅️ Leave Room", key="leave_room", use_container_width=True):
        st.session_state.current_page = 'lobby'
        st.session_state.room_code = ''
        st.session_state.players = []
        st.rerun()


def render_landing_page():
    """Render landing page (legacy - redirects to lobby)"""
    st.session_state.current_page = 'lobby'
    st.rerun()


def render_settings_page():
    """Render settings page"""
    st.markdown("""
    <div style="text-align: center; padding: 30px 20px;">
        <h2 style="color: var(--theme-gold); font-size: 36px; margin-bottom: 10px;">⚙️ Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🔄 Default Rounds</div>", unsafe_allow_html=True)
        num_rounds = st.selectbox("Rounds:", [3, 5, 7], index=0, key="settings_rounds")
        st.session_state.num_rounds = num_rounds
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>📏 Default Argument Length</div>", unsafe_allow_html=True)
        arg_length = st.selectbox("Length:", ["short", "medium", "long"], index=1, key="settings_length")
        st.session_state.argument_length = arg_length
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>⚖️ Default Judge Personality</div>", unsafe_allow_html=True)
        judge = st.selectbox("Judge:", ["balanced", "strict", "lenient"], index=0, key="settings_judge")
        st.session_state.judge_personality = judge
    
    with col2:
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🎨 Current Theme</div>", unsafe_allow_html=True)
        current_theme = get_active_theme()
        st.markdown(f"<div style='color: var(--theme-text-light); margin-bottom: 10px;'>{current_theme['name']}</div>", unsafe_allow_html=True)
        
        if st.button("🎨 Browse Themes", key="browse_themes", use_container_width=True):
            st.session_state.current_page = 'themes'
            st.rerun()
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>📚 Show Citations</div>", unsafe_allow_html=True)
        show_citations = st.checkbox("Show Citations", value=True, key="settings_citations")
        st.session_state.show_citations = show_citations
        
        st.markdown("<div style='color: var(--theme-text); font-weight: bold; margin-bottom: 10px;'>🎮 Default AI Difficulty</div>", unsafe_allow_html=True)
        difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard"], index=1, key="settings_difficulty")
        st.session_state.ai_difficulty = difficulty
    
    st.markdown("---")
    
    if st.button("💾 Save Settings", key="save_settings", use_container_width=True, type="primary"):
        st.success("Settings saved!")
        st.rerun()
    
    if st.button("⬅️ Back to Lobby", key="settings_back", use_container_width=True):
        st.session_state.current_page = 'lobby'
        st.rerun()


def render_themes_page():
    """Render premium theme gallery page with game-style UI using Streamlit components"""
    st.markdown("""
    <div style="text-align: center; padding: 20px 20px 40px 20px;">
        <h2 style="color: var(--theme-gold); font-size: 42px; margin-bottom: 10px; text-shadow: 0 0 20px rgba(255, 215, 0, 0.3);">🎨 Theme Gallery</h2>
        <p style="color: var(--theme-text-light); font-size: 18px; margin-bottom: 5px;">Choose your arena style</p>
        <p style="color: var(--theme-text-light); font-size: 14px; opacity: 0.7;">Click card to select</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme icons mapping
    theme_icons = {
        'midnight': '🌙',
        'royal': '👑'
    }
    
    # Display theme cards in responsive grid
    theme_keys = list(THEMES.keys())
    
    # Create grid with 2 columns for 2 themes
    cols = st.columns(2)
    
    for i, theme_key in enumerate(theme_keys):
        theme = THEMES[theme_key]
        is_active = st.session_state.get('theme', 'royal') == theme_key
        icon = theme_icons.get(theme_key, '🎨')
        
        with cols[i % 2]:
            # Create clickable card using Streamlit components
            with st.container():
                # Card container
                card_container = st.container()
                
                with card_container:
                    # Active badge
                    if is_active:
                        st.markdown(f"""
                        <div style="text-align: right; margin-bottom: 5px;">
                            <span style="background: linear-gradient(135deg, {theme['gold']}, {theme['accent']}); color: black; padding: 5px 12px; border-radius: 15px; font-size: 11px; font-weight: bold;">✓ Active</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Theme icon
                    st.markdown(f"""
                    <div style="text-align: center; font-size: 48px; margin-bottom: 10px; filter: drop-shadow(0 6px 12px rgba(0, 0, 0, 0.4));">
                        {icon}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Theme name
                    theme_name_display = theme['name'].split()[1] if len(theme['name'].split()) > 1 else theme['name']
                    st.markdown(f"""
                    <div style="text-align: center; font-size: 18px; font-weight: bold; margin-bottom: 18px; color: {theme['text']}; letter-spacing: 0.5px; text-transform: uppercase;">
                        {theme_name_display}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mini lobby preview
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {theme['background']}, {theme['sidebar']}); border-radius: 16px; padding: 16px; margin-top: 14px; border: 2px solid rgba(255, 255, 255, 0.2); box-shadow: inset 0 4px 12px rgba(0, 0, 0, 0.3);">
                        <div style="text-align: center; margin-bottom: 12px;">
                            <div style="font-size: 14px; font-weight: bold; letter-spacing: 2px; color: {theme['gold']}; margin-bottom: 4px;">⚔️ DebateBot</div>
                            <div style="font-size: 10px; opacity: 0.8; color: {theme['text_light']};">The Ultimate Debate Arena</div>
                        </div>
                        <div style="display: flex; gap: 8px; justify-content: center;">
                            <div style="background: linear-gradient(135deg, {theme['pro_color']}, {theme['success']}); padding: 10px 8px; border-radius: 12px; font-size: 9px; font-weight: bold; color: white; text-align: center; flex: 1; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4); display: flex; flex-direction: column; align-items: center; gap: 4px;">
                                <div style="font-size: 20px;">🤖</div>
                                <div style="font-size: 8px; text-transform: uppercase; letter-spacing: 0.5px;">AI vs AI</div>
                            </div>
                            <div style="background: linear-gradient(135deg, {theme['con_color']}, {theme['danger']}); padding: 10px 8px; border-radius: 12px; font-size: 9px; font-weight: bold; color: white; text-align: center; flex: 1; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4); display: flex; flex-direction: column; align-items: center; gap: 4px;">
                                <div style="font-size: 20px;">👤</div>
                                <div style="font-size: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Human vs AI</div>
                            </div>
                            <div style="background: linear-gradient(135deg, {theme['judge_color']}, {theme['info']}); padding: 10px 8px; border-radius: 12px; font-size: 9px; font-weight: bold; color: white; text-align: center; flex: 1; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4); display: flex; flex-direction: column; align-items: center; gap: 4px;">
                                <div style="font-size: 20px;">👥</div>
                                <div style="font-size: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Multiplayer</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Selection button
                button_label = f"Select {theme_name_display}" if not is_active else f"{theme_name_display} (Active)"
                button_type = "primary" if not is_active else "secondary"
                if st.button(button_label, key=f"select_{theme_key}", use_container_width=True, type=button_type):
                    theme_manager.set_theme(theme_key)
                    st.rerun()
    
    st.markdown("---")
    
    # Back button with premium styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⬅️ Back to Settings", key="themes_back", use_container_width=True):
            st.session_state.current_page = 'settings'
            st.rerun()


def render_help_page():
    """Render help/about page"""
    st.markdown("""
    <div style="text-align: center; padding: 30px 20px;">
        <h2 style="color: var(--theme-gold); font-size: 36px; margin-bottom: 10px;">❓ Help & About</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="gradient-card">
        <h3 style="color: var(--theme-success); margin-bottom: 15px;">🎮 How to Play</h3>
        <p style="color: var(--theme-text); line-height: 1.6;">
            <strong>AI vs AI:</strong> Watch two AI agents debate each other on any topic you choose.<br><br>
            <strong>Human vs AI:</strong> Challenge an AI to a debate. Choose your side (In Favour or Against) and argue your points.<br><br>
            <strong>Play with Friends:</strong> Create a room and invite friends to join using a room code, or join an existing room.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="gradient-card">
        <h3 style="color: var(--theme-danger); margin-bottom: 15px;">⚖️ How the Judge Works</h3>
        <p style="color: var(--theme-text); line-height: 1.6;">
            The Judge AI evaluates each argument based on logic, evidence, persuasiveness, clarity, and relevance. 
            After each round, scores are updated. At the end, the Judge declares a winner with detailed reasoning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="gradient-card">
        <h3 style="color: var(--theme-info); margin-bottom: 15px;">📏 Argument Lengths</h3>
        <p style="color: var(--theme-text); line-height: 1.6;">
            <strong>Short:</strong> 50-80 words, 3-5 concise sentences.<br>
            <strong>Medium:</strong> 100-150 words with explanation and one example.<br>
            <strong>Long:</strong> 180-250 words with multiple points and detailed reasoning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="gradient-card">
        <h3 style="color: var(--theme-warning); margin-bottom: 15px;">ℹ️ About DebateBot</h3>
        <p style="color: var(--theme-text); line-height: 1.6;">
            DebateBot is an AI-powered debate arena powered by LangGraph, Groq, and Tavily. 
            It uses advanced language models to generate intelligent arguments and evaluate debates fairly.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("⬅️ Back to Lobby", key="help_back", use_container_width=True):
        st.session_state.current_page = 'lobby'
        st.rerun()


def render_settings_sidebar():
    """Render settings sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #ffd700; margin-bottom: 5px;">⚙️ Settings</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Number of rounds
        st.markdown("<div style='color: white; font-weight: bold; margin-bottom: 10px;'>⚔️ Debate Settings</div>", unsafe_allow_html=True)
        num_rounds = st.selectbox("Number of Rounds:", [3, 5, 7], index=0, key="num_rounds_select")
        st.session_state.num_rounds = num_rounds
        
        # Argument length
        arg_length = st.selectbox("Argument Length:", ["short", "medium", "long"], index=1, key="arg_length_select")
        st.session_state.argument_length = arg_length
        
        # Judge personality
        judge_personality = st.selectbox("Judge Personality:", ["balanced", "strict", "lenient"], index=0, key="judge_personality_select")
        st.session_state.judge_personality = judge_personality
        
        st.markdown("---")
        
        # Citation toggle
        show_citations = st.checkbox("Show Citations", value=True, key="citation_checkbox")
        st.session_state.show_citations = show_citations
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Reset Settings", key="reset_settings", use_container_width=True):
            st.session_state.num_rounds = 3
            st.session_state.argument_length = 'medium'
            st.session_state.judge_personality = 'balanced'
            st.session_state.show_citations = True
            st.rerun()


def render_debate_header():
    """Render common debate arena header"""
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #ffd700; font-size: 36px; margin-bottom: 10px;">⚔️ Debate Arena</h1>
        <div style="background: #1a1a1a; border: 2px solid #ffd700; border-radius: 15px; padding: 15px 25px; display: inline-block; margin: 10px 0;">
            <span style="color: white; font-weight: bold; font-size: 18px;">Topic: {st.session_state.topic}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_debate_messages():
    """Render debate messages and citations"""
    # Message display
    st.markdown(f"""
    <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 20px; padding: 20px; min-height: 200px; max-height: 500px; overflow-y: auto;">
        <h3 style="color: white; margin-bottom: 15px;">💬 Live Debate</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Show ALL messages
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            render_message(msg, i)
    elif st.session_state.debate_complete and not st.session_state.show_transcript:
        st.markdown("""
        <div style="text-align: center; padding: 20px; color: #aaaaaa;">
            <p>Debate complete! Click "Show Debate Transcript" to view the full conversation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Citations
    if st.session_state.show_citations and st.session_state.citations:
        st.markdown("---")
        st.markdown("<div style='color: #ffd700; font-weight: bold; margin-bottom: 10px;'>📚 Citations</div>", unsafe_allow_html=True)
        for citation in st.session_state.citations:
            st.markdown(f"""
            <div style="background: #1a1a1a; border: 1px solid #333; border-left: 4px solid #ffd700; border-radius: 8px; padding: 12px 15px; margin: 10px 0;">
                <div style="color: white; font-weight: bold; font-size: 14px;">{citation.get('source', 'Unknown')}</div>
                <a href="{citation.get('url', '')}" target="_blank" style="color: #ffd700; text-decoration: none; font-size: 12px;">🔗 View Source</a>
            </div>
            """, unsafe_allow_html=True)


def render_judge_status():
    """Render judge status indicator"""
    if st.session_state.debate_active:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #1a1a1a; border: 2px solid #ffd700; border-radius: 10px;">
            <div style="color: #ffd700; font-size: 18px; font-weight: bold;">⚖️ Judge is observing the debate...</div>
        </div>
        """, unsafe_allow_html=True)


def render_debate_controls():
    """Render common debate completion controls (judge call, transcript, back button)"""
    if st.session_state.debate_complete:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("⚖️ CALL JUDGE", key="call_judge", use_container_width=True, type="primary"):
                with st.spinner("Judge is analyzing the debate..."):
                    st.session_state.judge_verdict = get_judge_verdict()
                st.session_state.current_page = 'verdict'
                st.rerun()
        
        if st.button("📜 Show Transcript", key="show_transcript", use_container_width=True):
            st.session_state.show_round_transcript = True
            st.rerun()
    
    # Round transcript expander
    if st.session_state.show_round_transcript:
        with st.expander("📜 Round Transcript", expanded=True):
            st.markdown("<h4 style='color: white; margin-bottom: 15px;'>Current Round Summary</h4>", unsafe_allow_html=True)
            if st.session_state.messages:
                round_start = st.session_state.current_round * 2
                round_messages = st.session_state.messages[round_start:round_start+2] if round_start < len(st.session_state.messages) else st.session_state.messages[-2:]
                for i, msg in enumerate(round_messages):
                    render_message(msg, i)
            
            if st.session_state.citations:
                st.markdown("<h4 style='color: #ffd700; margin: 15px 0;'>📚 Citations</h4>", unsafe_allow_html=True)
                for citation in st.session_state.citations[-3:]:
                    st.markdown(f"""
                    <div style="background: #1a1a1a; border: 1px solid #333; border-left: 4px solid #ffd700; border-radius: 8px; padding: 12px 15px; margin: 10px 0;">
                        <div style="color: white; font-weight: bold; font-size: 14px;">{citation.get('source', 'Unknown')}</div>
                        <a href="{citation.get('url', '')}" target="_blank" style="color: #ffd700; text-decoration: none; font-size: 12px;">🔗 View Source</a>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #1a1a1a; border: 2px solid #ffd700; border-radius: 15px; padding: 20px; margin: 15px 0;">
                <div style="color: #ffd700; font-weight: bold; font-size: 16px; margin-bottom: 15px; text-align: center;">📊 Round Scores</div>
                <div style="display: flex; justify-content: space-around; align-items: center;">
                    <div style="text-align: center;">
                        <div style="color: #00ff00; font-weight: bold; font-size: 24px;">{int(st.session_state.pro_score)}%</div>
                        <div style="color: white; font-size: 14px;">PRO</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #ff0000; font-weight: bold; font-size: 24px;">{int(st.session_state.con_score)}%</div>
                        <div style="color: white; font-size: 14px;">CON</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Back to lobby with confirmation
    if st.button("🏠 Back to Lobby", key="back_to_lobby", use_container_width=True):
        if st.session_state.debate_active or st.session_state.debate_complete:
            st.session_state.show_leave_confirmation = True
            st.rerun()
        else:
            st.session_state.current_page = 'lobby'
            st.rerun()
    
    # Show confirmation dialog
    if st.session_state.show_leave_confirmation:
        st.markdown("""
        <div style="background: #1a1a1a; border: 2px solid #ffd700; border-radius: 15px; padding: 20px; margin: 20px 0;">
            <h3 style="color: #ffd700; margin-bottom: 15px;">⚠️ Leave Debate?</h3>
            <p style="color: white; margin-bottom: 20px;">Are you sure you want to leave? This will reset the current debate and you will lose all progress.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("✅ Yes, Leave", key="confirm_leave", use_container_width=True, type="primary"):
                reset_debate_state()
                st.session_state.current_page = 'lobby'
                st.session_state.show_leave_confirmation = False
                st.rerun()
        with col2:
            if st.button("❌ Cancel", key="cancel_leave", use_container_width=True):
                st.session_state.show_leave_confirmation = False
                st.rerun()


def render_ai_vs_ai():
    """Render AI vs AI mode - both Pro and Con are AI agents"""
    render_debate_header()
    render_health_bars()
    render_progress_indicator()
    render_scoreboard()
    st.markdown("---")
    render_judge_status()
    render_debate_messages()
    st.markdown("---")
    
    # AI vs AI control buttons
    if st.session_state.debate_active:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Manual controls
            if st.session_state.current_turn == 'pro' and not st.session_state.pro_spoken_this_round:
                if st.button("▶️ Pro AI Turn", key="pro_ai_turn", use_container_width=True, type="primary"):
                    simulate_pro_turn()
                    st.rerun()
            elif st.session_state.current_turn == 'con' and not st.session_state.con_spoken_this_round:
                if st.button("▶️ Con AI Turn", key="con_ai_turn", use_container_width=True, type="primary"):
                    simulate_con_turn()
                    st.rerun()
            elif st.session_state.pro_spoken_this_round and st.session_state.con_spoken_this_round:
                if st.button("▶️ Next Round", key="next_round_ai_vs_ai", use_container_width=True, type="primary"):
                    st.session_state.current_round += 1
                    st.session_state.pro_spoken_this_round = False
                    st.session_state.con_spoken_this_round = False
                    st.session_state.current_turn = 'pro'
                    st.session_state.show_round_transcript = False
                    
                    if st.session_state.current_round > st.session_state.num_rounds:
                        st.session_state.debate_active = False
                        st.session_state.debate_complete = True
                        st.session_state.current_round = st.session_state.num_rounds
                    st.rerun()
            
            # Automate button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🤖 Automate Arguments", key="automate_ai_vs_ai", use_container_width=True):
                st.session_state.automating = True
                st.rerun()
            
            # Automation logic
            if st.session_state.get('automating', False) and st.session_state.debate_active:
                while st.session_state.current_round <= st.session_state.num_rounds:
                    if not st.session_state.pro_spoken_this_round:
                        simulate_pro_turn()
                    if not st.session_state.con_spoken_this_round:
                        simulate_con_turn()
                    
                    if st.session_state.pro_spoken_this_round and st.session_state.con_spoken_this_round:
                        if st.session_state.current_round >= st.session_state.num_rounds:
                            st.session_state.debate_active = False
                            st.session_state.debate_complete = True
                            st.session_state.automating = False
                            break
                        st.session_state.current_round += 1
                        st.session_state.pro_spoken_this_round = False
                        st.session_state.con_spoken_this_round = False
                        st.session_state.current_turn = 'pro'
                st.session_state.automating = False
                st.rerun()
    
    render_debate_controls()


def render_human_vs_ai():
    """Render Human vs AI mode - one human, one AI opponent"""
    render_debate_header()
    render_health_bars()
    render_progress_indicator()
    render_scoreboard()
    st.markdown("---")
    render_judge_status()
    render_debate_messages()
    st.markdown("---")
    
    # Human vs AI control buttons
    if st.session_state.debate_active:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            user_side = st.session_state.get('user_side', 'pro')
            
            # Determine whose turn it is
            if st.session_state.current_turn == 'pro' and not st.session_state.pro_spoken_this_round:
                if user_side == 'pro':
                    # Human's turn as Pro
                    user_argument = st.text_area("Your argument (Pro):", placeholder="Enter your argument...", key="human_pro_input", height=100)
                    if st.button("Submit Argument", key="submit_human_pro", use_container_width=True, type="primary"):
                        if user_argument and user_argument.strip():
                            st.session_state.messages.append(AIMessage(content=user_argument.strip(), name="Pro"))
                            st.session_state.pro_spoken_this_round = True
                            st.session_state.current_turn = 'con'
                            st.rerun()
                else:
                    # AI's turn as Pro
                    if st.button("▶️ Generate AI Argument (Pro)", key="ai_pro_turn", use_container_width=True, type="primary"):
                        simulate_pro_turn()
                        st.rerun()
            
            elif st.session_state.current_turn == 'con' and not st.session_state.con_spoken_this_round:
                if user_side == 'con':
                    # Human's turn as Con
                    user_argument = st.text_area("Your argument (Con):", placeholder="Enter your argument...", key="human_con_input", height=100)
                    if st.button("Submit Argument", key="submit_human_con", use_container_width=True, type="primary"):
                        if user_argument and user_argument.strip():
                            st.session_state.messages.append(AIMessage(content=user_argument.strip(), name="Con"))
                            st.session_state.con_spoken_this_round = True
                            st.session_state.current_turn = 'pro'
                            st.rerun()
                else:
                    # AI's turn as Con
                    if st.button("▶️ Generate AI Argument (Con)", key="ai_con_turn", use_container_width=True, type="primary"):
                        simulate_con_turn()
                        st.rerun()
            
            elif st.session_state.pro_spoken_this_round and st.session_state.con_spoken_this_round:
                if st.button("▶️ Next Round", key="next_round_human_vs_ai", use_container_width=True, type="primary"):
                    st.session_state.current_round += 1
                    st.session_state.pro_spoken_this_round = False
                    st.session_state.con_spoken_this_round = False
                    st.session_state.current_turn = 'pro'
                    st.session_state.show_round_transcript = False
                    
                    if st.session_state.current_round > st.session_state.num_rounds:
                        st.session_state.debate_active = False
                        st.session_state.debate_complete = True
                        st.session_state.current_round = st.session_state.num_rounds
                    st.rerun()
    
    render_debate_controls()


def render_play_with_friends():
    """Render Play with Friends mode - human players only"""
    render_debate_header()
    render_health_bars()
    render_progress_indicator()
    render_scoreboard()
    st.markdown("---")
    render_judge_status()
    render_debate_messages()
    st.markdown("---")
    
    # Play with Friends control buttons
    if st.session_state.debate_active:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state.current_turn == 'pro' and not st.session_state.pro_spoken_this_round:
                pro_argument = st.text_area("Pro's argument:", placeholder="Enter Pro's argument...", key="multiplayer_pro_input", height=100)
                if st.button("Submit Pro Argument", key="submit_multiplayer_pro", use_container_width=True, type="primary"):
                    if pro_argument and pro_argument.strip():
                        st.session_state.messages.append(AIMessage(content=pro_argument.strip(), name="Pro"))
                        st.session_state.pro_spoken_this_round = True
                        st.session_state.current_turn = 'con'
                        st.rerun()
            
            elif st.session_state.current_turn == 'con' and not st.session_state.con_spoken_this_round:
                con_argument = st.text_area("Con's argument:", placeholder="Enter Con's argument...", key="multiplayer_con_input", height=100)
                if st.button("Submit Con Argument", key="submit_multiplayer_con", use_container_width=True, type="primary"):
                    if con_argument and con_argument.strip():
                        st.session_state.messages.append(AIMessage(content=con_argument.strip(), name="Con"))
                        st.session_state.con_spoken_this_round = True
                        st.session_state.current_turn = 'pro'
                        st.rerun()
            
            elif st.session_state.pro_spoken_this_round and st.session_state.con_spoken_this_round:
                if st.button("▶️ Next Round", key="next_round_multiplayer", use_container_width=True, type="primary"):
                    st.session_state.current_round += 1
                    st.session_state.pro_spoken_this_round = False
                    st.session_state.con_spoken_this_round = False
                    st.session_state.current_turn = 'pro'
                    st.session_state.show_round_transcript = False
                    
                    if st.session_state.current_round > st.session_state.num_rounds:
                        st.session_state.debate_active = False
                        st.session_state.debate_complete = True
                        st.session_state.current_round = st.session_state.num_rounds
                    st.rerun()
    
    render_debate_controls()


def render_debate_arena():
    """Render debate arena - router for different game modes"""
    game_mode = st.session_state.get('game_mode', 'ai_vs_ai')
    
    if game_mode == 'ai_vs_ai':
        render_ai_vs_ai()
    elif game_mode == 'human_vs_ai':
        render_human_vs_ai()
    elif game_mode == 'multiplayer':
        render_play_with_friends()
    else:
        render_ai_vs_ai()  # Default to AI vs AI


def render_verdict_page():
    """Render judge verdict page"""
    game_mode = st.session_state.get('game_mode', 'ai_vs_ai')
    
    # For AI vs AI, read from session state storage
    if game_mode == 'ai_vs_ai' and 'ai_vs_ai_judge_result' in st.session_state:
        verdict = st.session_state['ai_vs_ai_judge_result']
        print("DEBUG: Reading AI vs AI verdict from session_state")
    else:
        verdict = st.session_state.judge_verdict
    
    # Set title based on game mode
    if game_mode == 'human_vs_ai':
        title = "🏆 Human vs AI Verdict"
    else:
        title = "⚖️ Judge's Verdict"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 30px 20px;">
        <h1 style="color: #ffd700; font-size: 42px; margin-bottom: 10px;">{title}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Winner announcement
    winner = verdict.get('winner', 'Draw')
    if 'Human' in winner or 'Pro' in winner:
        winner_color = '#00ff00'
        winner_emoji = "🏆"
    elif 'AI' in winner or 'Con' in winner:
        winner_color = '#ff0000'
        winner_emoji = "🏆"
    else:
        winner_color = '#ffd700'
        winner_emoji = "🤝"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {winner_color}22, {winner_color}44); border: 3px solid {winner_color}; border-radius: 20px; padding: 30px; text-align: center; margin: 20px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
        <div style="font-size: 60px; margin-bottom: 15px;">{winner_emoji}</div>
        <div style="color: white; font-size: 18px; margin-bottom: 10px;">THE WINNER IS</div>
        <div style="color: {winner_color}; font-size: 36px; font-weight: bold;">{winner.upper()}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Final score
    st.markdown(f"""
    <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 15px; padding: 20px; text-align: center; margin: 20px 0;">
        <div style="color: white; font-weight: bold; font-size: 18px; margin-bottom: 10px;">Final Score</div>
        <div style="color: #ffd700; font-size: 28px; font-weight: bold;">{verdict.get('final_score', 'N/A')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Final health bars
    st.markdown("<h3 style='text-align: center; margin: 20px 0;'>Final Persuasion Status</h3>", unsafe_allow_html=True)
    render_health_bars()
    
    st.markdown("---")
    
    # Detailed analysis
    col1, col2 = st.columns(2)
    
    # Set labels based on game mode
    if game_mode == 'human_vs_ai':
        user_side = st.session_state.get('user_side', 'pro')
        if user_side == 'pro':
            pro_label = "🟢 Human Strengths"
            con_label = "🔴 AI Strengths"
        else:
            pro_label = "🟢 AI Strengths"
            con_label = "🔴 Human Strengths"
    else:
        pro_label = "🟢 Pro Strengths"
        con_label = "🔴 Con Strengths"
    
    with col1:
        st.markdown(f"""
        <div style="background: #1a1a1a; border: 2px solid #00ff00; border-radius: 15px; padding: 20px; margin: 10px 0;">
            <h3 style="color: #00ff00; margin-bottom: 15px;">{pro_label}</h3>
            <p style="color: white; line-height: 1.6;">{verdict.get('pro_strengths', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: #1a1a1a; border: 2px solid #ff0000; border-radius: 15px; padding: 20px; margin: 10px 0;">
            <h3 style="color: #ff0000; margin-bottom: 15px;">{con_label}</h3>
            <p style="color: white; line-height: 1.6;">{verdict.get('con_strengths', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Best moments
    st.markdown(f"""
    <div style="background: #1a1a1a; border: 2px solid #ffd700; border-radius: 15px; padding: 25px; margin: 20px 0;">
        <h3 style="color: #ffd700; margin-bottom: 20px;">⭐ Best Moments</h3>
        <div style="margin: 15px 0;">
            <div style="color: white; font-weight: bold; margin-bottom: 8px;">🎯 Best Argument</div>
            <div style="color: #aaaaaa; line-height: 1.6;">{verdict.get('best_argument', 'N/A')}</div>
        </div>
        <div style="margin: 15px 0;">
            <div style="color: white; font-weight: bold; margin-bottom: 8px;">💥 Best Rebuttal</div>
            <div style="color: #aaaaaa; line-height: 1.6;">{verdict.get('best_rebuttal', 'N/A')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Final reasoning
    st.markdown(f"""
    <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 15px; padding: 25px; margin: 20px 0;">
        <h3 style="color: white; margin-bottom: 15px;">📝 Final Reasoning</h3>
        <p style="color: #aaaaaa; line-height: 1.8; font-size: 16px;">{verdict.get('final_reasoning', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏠 New Debate", key="new_debate", use_container_width=True):
            reset_debate_state()
            st.session_state.current_page = 'landing'
            st.rerun()
    
    with col2:
        if st.button("📜 View History", key="view_history_btn", use_container_width=True):
            st.session_state.current_page = 'history'
            st.rerun()
    
    with col3:
        if st.button("💾 Save Debate", key="save_debate", use_container_width=True):
            save_debate_to_history()
            st.success("Debate saved to history!")
    
    with col4:
        if st.button("📥 Export PDF", key="export_pdf", use_container_width=True):
            try:
                pdf_file = export_debate_to_pdf()
                with open(pdf_file, 'rb') as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name=os.path.basename(pdf_file),
                        mime="application/pdf",
                        key="download_pdf"
                    )
            except Exception as e:
                st.error(f"Error exporting PDF: {str(e)}")


def render_history_page():
    """Render debate history page"""
    st.markdown("""
    <div style="text-align: center; padding: 30px 20px;">
        <h1 style="color: #ffd700; font-size: 42px; margin-bottom: 10px;">📜 Debate History</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load history
    history = load_debate_history()
    
    # Search
    search_query = st.text_input("🔍 Search debates...", placeholder="Search by topic...", key="history_search")
    
    if search_query:
        history = [d for d in history if search_query.lower() in d.get('topic', '').lower()]
    
    if not history:
        st.info("No debates found. Start a new debate to see it here!")
        if st.button("🏠 Go to Landing", key="go_to_landing", use_container_width=True):
            st.session_state.current_page = 'landing'
            st.rerun()
        return
    
    # Display debates
    for debate in history:
        with st.container():
            st.markdown(f"""
            <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 15px; padding: 20px; margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                    <div>
                        <div style="color: white; font-weight: bold; font-size: 18px; margin-bottom: 5px;">{debate.get('topic', 'Untitled')}</div>
                        <div style="color: #aaaaaa; font-size: 14px;">{debate.get('timestamp', '')}</div>
                    </div>
                    <div style="background: #ffd70022; border: 1px solid #ffd700; border-radius: 10px; padding: 5px 15px;">
                        <div style="color: #ffd700; font-weight: bold; font-size: 12px;">WINNER</div>
                        <div style="color: #ffd700; font-weight: bold; font-size: 14px;">{debate.get('verdict', {}).get('winner', 'N/A')}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👁️ View", key=f"view_{debate['id']}", use_container_width=True):
                    # Load debate into session state for viewing
                    st.session_state.topic = debate.get('topic', '')
                    st.session_state.messages = [AIMessage(content=msg) for msg in debate.get('messages', [])]
                    st.session_state.citations = debate.get('citations', [])
                    st.session_state.judge_verdict = debate.get('verdict', {})
                    st.session_state.pro_score = debate.get('pro_score', 50.0)
                    st.session_state.con_score = debate.get('con_score', 50.0)
                    st.session_state.show_transcript = True
                    st.session_state.current_page = 'verdict'
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{debate['id']}", use_container_width=True):
                    delete_debate(debate['id'])
                    st.success("Debate deleted!")
                    st.rerun()
    
    # Back button
    st.markdown("---")
    if st.button("🏠 Back to Lobby", key="back_to_lobby_history", use_container_width=True):
        st.session_state.current_page = 'lobby'
        st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application router"""
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Route to appropriate page
    current_page = st.session_state.current_page
    
    if current_page == 'lobby':
        render_lobby()
    elif current_page == 'ai_vs_ai_config':
        render_ai_vs_ai_config()
    elif current_page == 'human_vs_ai_config':
        render_human_vs_ai_config()
    elif current_page == 'multiplayer_lobby':
        render_multiplayer_lobby()
    elif current_page == 'room_lobby':
        render_room_lobby()
    elif current_page == 'debate':
        render_debate_arena()
    elif current_page == 'verdict':
        render_verdict_page()
    elif current_page == 'history':
        render_history_page()
    elif current_page == 'settings':
        render_settings_page()
    elif current_page == 'themes':
        render_themes_page()
    elif current_page == 'help':
        render_help_page()
    else:
        st.session_state.current_page = 'lobby'
        render_lobby()


if __name__ == "__main__":
    main()