"""
DebateBot: Two Sides and a Judge
AI-powered debate arena with LangGraph, Groq, and Tavily
"""

import streamlit as st
import os
import json
import time
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

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


def create_pro_agent():
    """Create the Pro agent with Groq LLM"""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=GROQ_API_KEY
    )
    
    system_prompt = """You are arguing IN FAVOR of the given topic in a formal debate.

Rules:
1. Always support the topic with strong, logical arguments
2. Use evidence and facts to back up your claims
3. Be respectful but firm in your position
4. Address counter-arguments directly when responding
5. Keep your arguments concise (2-3 paragraphs maximum)
6. Use the search tool to find real evidence and citations
7. Always cite your sources when making factual claims
8. NEVER identify yourself as "a pro agent" or "as the pro agent" - simply state your arguments directly
9. Do not use phrases like "As a pro agent" or "Speaking as the pro side"

Your goal is to persuade the judge that your position is correct through superior reasoning and evidence."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    return prompt | llm


def create_con_agent():
    """Create the Con agent with Groq LLM"""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=GROQ_API_KEY
    )
    
    system_prompt = """You are arguing AGAINST the given topic in a formal debate.

Rules:
1. Always oppose the topic with strong, logical arguments
2. Use evidence and facts to back up your claims
3. Be respectful but firm in your position
4. Address the opponent's arguments directly
5. Keep your rebuttals concise (2-3 paragraphs maximum)
6. Use the search tool to find real evidence and citations
7. Always cite your sources when making factual claims
8. NEVER identify yourself as "a con agent" or "as the con agent" - simply state your rebuttals directly
9. Do not use phrases like "As a con agent" or "Speaking as the con side"

Your goal is to persuade the judge that your position is correct through superior reasoning and evidence."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    return prompt | llm


def create_judge_agent():
    """Create the Judge agent with Groq LLM"""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )
    
    system_prompt = """You are an impartial JUDGE in a formal debate. Your role is to evaluate the arguments from both sides and declare a winner.

Evaluation Criteria:
1. Logical consistency and coherence
2. Quality and relevance of evidence
3. Strength of counter-arguments
4. Persuasiveness and clarity
5. Use of citations and sources

Provide your verdict in the following format:
- Winner: [Pro Agent or Con Agent]
- Final Score: [Pro X - Con Y]
- Best Argument: [Describe the strongest argument]
- Best Rebuttal: [Describe the strongest rebuttal]
- Pro Strengths: [List key strengths of Pro]
- Con Strengths: [List key strengths of Con]
- Final Reasoning: [Provide detailed reasoning for your decision]

Be fair, objective, and thorough in your evaluation."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    return prompt | llm


def create_search_tool():
    """Create Tavily search tool for evidence retrieval"""
    return TavilySearchResults(
        max_results=3,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
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
# CUSTOM CSS
# ============================================================================

def load_custom_css():
    """Load custom CSS for fighting game theme"""
    css = """
    <style>
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Health bar animations */
    @keyframes healthPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .health-pulse {
        animation: healthPulse 2s infinite;
    }
    
    /* Winner glow animation */
    @keyframes winnerGlow {
        0%, 100% { box-shadow: 0 8px 16px rgba(0,0,0,0.3); }
        50% { box-shadow: 0 8px 30px rgba(255, 215, 0, 0.5); }
    }
    
    .winner-glow {
        animation: winnerGlow 2s infinite;
    }
    
    /* Message slide animation */
    @keyframes messageSlide {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .message-slide {
        animation: messageSlide 0.3s ease-out;
    }
    
    /* Typing indicator */
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
    
    .typing-dot {
        animation: typing 1.4s infinite ease-in-out;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #ffd700, #ff8c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #ffd700;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #ff8c00;
    }
    
    /* Button hover effects */
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Hide footer */
    footer {
        visibility: hidden;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'landing'
    
    if 'topic' not in st.session_state:
        st.session_state.topic = ''
    
    if 'num_rounds' not in st.session_state:
        st.session_state.num_rounds = 3
    
    if 'argument_length' not in st.session_state:
        st.session_state.argument_length = 'medium'
    
    if 'judge_personality' not in st.session_state:
        st.session_state.judge_personality = 'balanced'
    
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    
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


def reset_debate_state():
    """Reset debate state for new debate"""
    st.session_state.debate_active = False
    st.session_state.debate_complete = False
    st.session_state.current_round = 0
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
        role = "PRO AGENT" if i % 2 == 0 else "CON AGENT"
        color = "#00ff00" if i % 2 == 0 else "#ff0000"
        story.append(Paragraph(f"<b>{role}:</b>", styles['Normal']))
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
        
        # Create message history with ALL previous messages for debate memory
        messages = [
            SystemMessage(content=f"You are debating the topic: {topic}. This is round {round_num + 1} of {st.session_state.num_rounds}. Review all previous arguments and rebuttals before responding.")
        ]
        
        # Add ALL previous messages for full debate memory
        for msg in st.session_state.messages:
            messages.append(msg)
        
        # Add current round instruction
        messages.append(HumanMessage(content=f"Provide your argument for round {round_num + 1}. Support the topic with evidence. Reference previous arguments when relevant."))
        
        response = pro_agent.invoke({"messages": messages})
        return response.content
    except Exception as e:
        return f"Error generating Pro argument: {str(e)}"


def get_con_rebuttal(topic: str, round_num: int) -> str:
    """Get Con agent's rebuttal using Groq"""
    try:
        con_agent = create_con_agent()
        
        # Create message history with ALL previous messages for debate memory
        messages = [
            SystemMessage(content=f"You are debating the topic: {topic}. This is round {round_num + 1} of {st.session_state.num_rounds}. Review all previous arguments and rebuttals before responding.")
        ]
        
        # Add ALL previous messages for full debate memory
        for msg in st.session_state.messages:
            messages.append(msg)
        
        # Add current round instruction
        messages.append(HumanMessage(content=f"Provide your rebuttal for round {round_num + 1}. Oppose the topic with evidence. Reference previous arguments when relevant."))
        
        response = con_agent.invoke({"messages": messages})
        return response.content
    except Exception as e:
        return f"Error generating Con rebuttal: {str(e)}"


def get_judge_verdict() -> Dict[str, str]:
    """Get judge's verdict using Groq"""
    try:
        judge_agent = create_judge_agent()
        
        # Create message history with full debate
        messages = [
            SystemMessage(content=f"You are judging a debate on the topic: {st.session_state.topic}")
        ]
        
        # Add all debate messages
        for msg in st.session_state.messages:
            messages.append(msg)
        
        # Add verdict request
        messages.append(HumanMessage(content="Provide your final verdict based on the debate above."))
        
        response = judge_agent.invoke({"messages": messages})
        verdict_text = response.content
        
        # Parse verdict (simple parsing - could be improved)
        verdict = {
            'winner': 'Draw',
            'final_score': f"Pro: {int(st.session_state.pro_score)} - Con: {int(st.session_state.con_score)}",
            'best_argument': 'Not specified',
            'best_rebuttal': 'Not specified',
            'pro_strengths': 'Not specified',
            'con_strengths': 'Not specified',
            'final_reasoning': verdict_text
        }
        
        # Try to extract structured information
        if 'Winner:' in verdict_text:
            winner_line = [line for line in verdict_text.split('\n') if 'Winner:' in line]
            if winner_line:
                verdict['winner'] = winner_line[0].split('Winner:')[1].strip()
        
        if 'Final Score:' in verdict_text:
            score_line = [line for line in verdict_text.split('\n') if 'Final Score:' in line]
            if score_line:
                verdict['final_score'] = score_line[0].split('Final Score:')[1].strip()
        
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
    """Simulate Pro agent's turn"""
    topic = st.session_state.topic
    round_num = st.session_state.current_round
    
    # Pro Agent's turn
    with st.empty():
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="color: #00ff00; font-size: 24px; font-weight: bold;">
                🟢 Pro Agent is thinking...
            </div>
            <div style="display: flex; justify-content: center; gap: 8px; margin-top: 10px;">
                <div class="typing-dot" style="width: 10px; height: 10px; background: #00ff00; border-radius: 50%;"></div>
                <div class="typing-dot" style="width: 10px; height: 10px; background: #00ff00; border-radius: 50%;"></div>
                <div class="typing-dot" style="width: 10px; height: 10px; background: #00ff00; border-radius: 50%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(2)
    
    pro_argument = get_pro_argument(topic, round_num)
    st.session_state.messages.append(AIMessage(content=pro_argument, name="Pro"))
    st.session_state.pro_spoken_this_round = True
    st.session_state.current_turn = 'con'
    
    # Search for citations
    if st.session_state.show_citations:
        citations = search_tavily(f"{topic} evidence support")
        st.session_state.citations.extend(citations)
    
    # Update scores (simple random for demo - could be improved with actual evaluation)
    import random
    pro_change = random.uniform(-5, 10)
    con_change = random.uniform(-10, 5)
    st.session_state.pro_score = max(0, min(100, st.session_state.pro_score + pro_change))
    st.session_state.con_score = max(0, min(100, st.session_state.con_score + con_change))


def simulate_con_turn():
    """Simulate Con agent's turn"""
    topic = st.session_state.topic
    round_num = st.session_state.current_round
    
    # Con Agent's turn
    with st.empty():
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="color: #ff0000; font-size: 24px; font-weight: bold;">
                🔴 Con Agent is thinking...
            </div>
            <div style="display: flex; justify-content: center; gap: 8px; margin-top: 10px;">
                <div class="typing-dot" style="width: 10px; height: 10px; background: #ff0000; border-radius: 50%;"></div>
                <div class="typing-dot" style="width: 10px; height: 10px; background: #ff0000; border-radius: 50%;"></div>
                <div class="typing-dot" style="width: 10px; height: 10px; background: #ff0000; border-radius: 50%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(2)
    
    con_rebuttal = get_con_rebuttal(topic, round_num)
    st.session_state.messages.append(AIMessage(content=con_rebuttal, name="Con"))
    st.session_state.con_spoken_this_round = True
    st.session_state.current_turn = 'pro'
    
    # Search for citations
    if st.session_state.show_citations:
        citations = search_tavily(f"{topic} evidence oppose")
        st.session_state.citations.extend(citations)
    
    # Update scores
    pro_change = random.uniform(-10, 5)
    con_change = random.uniform(-5, 10)
    st.session_state.pro_score = max(0, min(100, st.session_state.pro_score + pro_change))
    st.session_state.con_score = max(0, min(100, st.session_state.con_score + con_change))


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_health_bars():
    """Render health/persuasion bars"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 10px;">
            <div style="font-weight: bold; color: #00ff00; font-size: 18px;">PRO AGENT</div>
            <div style="background: #1a1a1a; border: 2px solid #00ff00; border-radius: 10px; height: 30px; margin: 10px 0; position: relative; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #00ff00, #32cd32); height: 100%; width: {st.session_state.pro_score}%; transition: width 0.5s ease;"></div>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-weight: bold; text-shadow: 1px 1px 2px black;">{int(st.session_state.pro_score)}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; margin: 15px 0;">
            <div style="background: #ffd700; color: black; padding: 10px 20px; border-radius: 20px; font-weight: bold; font-size: 20px; display: inline-block; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                ROUND {st.session_state.current_round + 1} / {st.session_state.num_rounds}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 10px;">
            <div style="font-weight: bold; color: #ff0000; font-size: 18px;">CON AGENT</div>
            <div style="background: #1a1a1a; border: 2px solid #ff0000; border-radius: 10px; height: 30px; margin: 10px 0; position: relative; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #ff6b6b, #ff0000); height: 100%; width: {st.session_state.con_score}%; transition: width 0.5s ease;"></div>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-weight: bold; text-shadow: 1px 1px 2px black;">{int(st.session_state.con_score)}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_progress_indicator():
    """Render debate progress indicator"""
    progress = (st.session_state.current_round / st.session_state.num_rounds) * 100
    st.markdown(f"""
    <div style="margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: white; font-weight: bold;">Debate Progress</span>
            <span style="color: #ffd700; font-weight: bold;">{int(progress)}%</span>
        </div>
        <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 10px; height: 12px; position: relative; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #ffd700, #ff8c00); height: 100%; width: {progress}%; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_message(message: AIMessage, index: int):
    """Render a single debate message"""
    is_pro = index % 2 == 0
    bg_color = '#1b4d1e' if is_pro else '#4a1a1a'
    border_color = '#00ff00' if is_pro else '#ff0000'
    align = 'left' if is_pro else 'right'
    agent_name = 'PRO AGENT' if is_pro else 'CON AGENT'
    
    st.markdown(f"""
    <div style="margin: 15px 0;">
        <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 15px; padding: 15px 20px; max-width: 80%; margin: {'0 auto 0 0' if align == 'left' else '0 0 0 auto'}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: {border_color}; font-weight: bold; font-size: 14px; margin-bottom: 10px;">{agent_name}</div>
            <div style="color: white; line-height: 1.6;">{message.content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_scoreboard():
    """Render scoreboard"""
    leader = "PRO" if st.session_state.pro_score > st.session_state.con_score else ("CON" if st.session_state.con_score > st.session_state.pro_score else "TIE")
    leader_color = "#00ff00" if leader == "PRO" else "#ff0000" if leader == "CON" else "#ffd700"
    
    st.markdown(f"""
    <div style="background: #1a1a1a; border: 2px solid #ffd700; border-radius: 15px; padding: 20px; margin: 15px 0;">
        <div style="color: #ffd700; font-weight: bold; font-size: 16px; margin-bottom: 15px; text-align: center;">📊 SCOREBOARD</div>
        <div style="display: flex; justify-content: space-around; align-items: center;">
            <div style="text-align: center;">
                <div style="color: #00ff00; font-weight: bold; font-size: 24px;">{int(st.session_state.pro_score)}%</div>
                <div style="color: white; font-size: 14px;">PRO</div>
            </div>
            <div style="text-align: center;">
                <div style="color: {leader_color}; font-weight: bold; font-size: 18px;">{leader}</div>
                <div style="color: #ffd700; font-size: 14px;">LEADER</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #ff0000; font-weight: bold; font-size: 24px;">{int(st.session_state.con_score)}%</div>
                <div style="color: white; font-size: 14px;">CON</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGES
# ============================================================================

def render_landing_page():
    """Render landing page"""
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="color: #ffd700; font-size: 48px; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            ⚔️ DebateBot
        </h1>
        <h2 style="color: white; font-size: 24px; margin-bottom: 30px; font-weight: normal;">
            Two Sides and a Judge
        </h2>
        <p style="color: #aaaaaa; font-size: 16px; max-width: 600px; margin: 0 auto 40px auto; line-height: 1.6;">
            Watch AI agents battle in intellectual combat. Pro and Con agents debate 
            controversial topics while a Judge evaluates their arguments and declares the winner.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Topic input
    st.markdown("""
    <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 20px; padding: 30px; margin: 20px 0;">
        <h3 style="color: white; margin-bottom: 20px;">🎯 Choose Your Debate Topic</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if 'input_topic' not in st.session_state:
        st.session_state.input_topic = ''
    
    topic = st.text_input(
        "Enter a debate topic:",
        placeholder="e.g., Should AI be regulated?",
        value=st.session_state.input_topic,
        key="topic_input",
        label_visibility="collapsed",
        on_change=lambda: setattr(st.session_state, 'input_topic', st.session_state.topic_input)
    )
    
    # Suggested topics
    suggested_topics = [
        "Should artificial intelligence be regulated?",
        "Is remote work better than office work?",
        "Should social media companies be held liable for content?",
        "Is universal basic income feasible?",
        "Should college education be free?",
        "Is cryptocurrency the future of finance?"
    ]
    
    st.markdown("<div style='color: #aaaaaa; margin-bottom: 10px;'>💡 Suggested topics:</div>", unsafe_allow_html=True)
    
    cols = st.columns(2)
    for i, suggested_topic in enumerate(suggested_topics[:6]):
        with cols[i % 2]:
            if st.button(suggested_topic, key=f"suggest_{i}", use_container_width=True):
                st.session_state.input_topic = suggested_topic
                st.rerun()
    
    st.markdown("---")
    
    # Start debate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⚔️ START DEBATE", key="start_debate", use_container_width=True, type="primary"):
            if topic and topic.strip():
                st.session_state.topic = topic.strip()
                st.session_state.current_page = 'debate'
                reset_debate_state()
                st.session_state.debate_active = True
                st.rerun()
            else:
                st.error("Please enter a debate topic to start.")
    
    # View history button
    if st.button("📜 View Debate History", key="view_history", use_container_width=True):
        st.session_state.current_page = 'history'
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


def render_debate_arena():
    """Render debate arena"""
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #ffd700; font-size: 36px; margin-bottom: 10px;">⚔️ Debate Arena</h1>
        <div style="background: #1a1a1a; border: 2px solid #ffd700; border-radius: 15px; padding: 15px 25px; display: inline-block; margin: 10px 0;">
            <span style="color: white; font-weight: bold; font-size: 18px;">Topic: {st.session_state.topic}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Health bars
    render_health_bars()
    
    # Progress indicator
    render_progress_indicator()
    
    # Scoreboard
    render_scoreboard()
    
    st.markdown("---")
    
    # Judge status
    if st.session_state.debate_active:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #1a1a1a; border: 2px solid #ffd700; border-radius: 10px;">
            <div style="color: #ffd700; font-size: 18px; font-weight: bold;">⚖️ Judge is observing the debate...</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Message display (show ALL messages - WhatsApp style)
    st.markdown(f"""
    <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 20px; padding: 20px; min-height: 200px; max-height: 500px; overflow-y: auto;">
        <h3 style="color: white; margin-bottom: 15px;">💬 Live Debate</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Show ALL messages (preserve history)
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            render_message(msg, i)
    elif st.session_state.debate_complete and not st.session_state.show_transcript:
        st.markdown("""
        <div style="text-align: center; padding: 20px; color: #aaaaaa;">
            <p>Debate complete! Click "Show Debate Transcript" to view the full conversation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Citations (show all, not just recent)
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
    
    st.markdown("---")
    
    # Control buttons
    if st.session_state.debate_active:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state.current_turn == 'pro' and not st.session_state.pro_spoken_this_round:
                if st.button("▶️ Next Turn", key="next_turn_pro", use_container_width=True, type="primary"):
                    simulate_pro_turn()
                    st.rerun()
            elif st.session_state.current_turn == 'con' and not st.session_state.con_spoken_this_round:
                if st.button("▶️ Next Turn", key="next_turn_con", use_container_width=True, type="primary"):
                    simulate_con_turn()
                    st.rerun()
            elif st.session_state.pro_spoken_this_round and st.session_state.con_spoken_this_round:
                if st.button("▶️ Next Round", key="next_round", use_container_width=True, type="primary"):
                    st.session_state.current_round += 1
                    st.session_state.pro_spoken_this_round = False
                    st.session_state.con_spoken_this_round = False
                    st.session_state.current_turn = 'pro'
                    st.session_state.show_round_transcript = False
                    
                    # Check if debate is complete
                    if st.session_state.current_round >= st.session_state.num_rounds:
                        st.session_state.debate_active = False
                        st.session_state.debate_complete = True
                    st.rerun()
            else:
                st.info("Waiting for next action...")
    elif st.session_state.debate_complete:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("⚖️ CALL JUDGE", key="call_judge", use_container_width=True, type="primary"):
                with st.spinner("Judge is analyzing the debate..."):
                    st.session_state.judge_verdict = get_judge_verdict()
                st.session_state.current_page = 'verdict'
                st.rerun()
        
        # Show transcript button
        if st.button("📜 Show Transcript", key="show_transcript", use_container_width=True):
            st.session_state.show_round_transcript = True
            st.rerun()
    
    # Round transcript expander
    if st.session_state.show_round_transcript:
        with st.expander("📜 Round Transcript", expanded=True):
            st.markdown("<h4 style='color: white; margin-bottom: 15px;'>Current Round Summary</h4>", unsafe_allow_html=True)
            if st.session_state.messages:
                # Show messages from current round
                round_start = st.session_state.current_round * 2
                round_messages = st.session_state.messages[round_start:round_start+2] if round_start < len(st.session_state.messages) else st.session_state.messages[-2:]
                for i, msg in enumerate(round_messages):
                    render_message(msg, i)
            
            # Show citations
            if st.session_state.citations:
                st.markdown("<h4 style='color: #ffd700; margin: 15px 0;'>📚 Citations</h4>", unsafe_allow_html=True)
                for citation in st.session_state.citations[-3:]:
                    st.markdown(f"""
                    <div style="background: #1a1a1a; border: 1px solid #333; border-left: 4px solid #ffd700; border-radius: 8px; padding: 12px 15px; margin: 10px 0;">
                        <div style="color: white; font-weight: bold; font-size: 14px;">{citation.get('source', 'Unknown')}</div>
                        <a href="{citation.get('url', '')}" target="_blank" style="color: #ffd700; text-decoration: none; font-size: 12px;">🔗 View Source</a>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show scores
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
    
    # Back to landing with confirmation
    if st.button("🏠 Back to Landing Page", key="back_to_landing", use_container_width=True):
        if st.session_state.debate_active or st.session_state.debate_complete:
            if st.confirm("Are you sure you want to leave? This will reset the current debate."):
                reset_debate_state()
                st.session_state.current_page = 'landing'
                st.rerun()
        else:
            st.session_state.current_page = 'landing'
            st.rerun()


def render_verdict_page():
    """Render judge verdict page"""
    verdict = st.session_state.judge_verdict
    
    st.markdown("""
    <div style="text-align: center; padding: 30px 20px;">
        <h1 style="color: #ffd700; font-size: 42px; margin-bottom: 10px;">⚖️ Judge's Verdict</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Winner announcement
    winner = verdict.get('winner', 'Draw')
    if 'Pro' in winner:
        winner_color = '#00ff00'
        winner_emoji = "🏆"
    elif 'Con' in winner:
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
    
    with col1:
        st.markdown(f"""
        <div style="background: #1a1a1a; border: 2px solid #00ff00; border-radius: 15px; padding: 20px; margin: 10px 0;">
            <h3 style="color: #00ff00; margin-bottom: 15px;">🟢 Pro Strengths</h3>
            <p style="color: white; line-height: 1.6;">{verdict.get('pro_strengths', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: #1a1a1a; border: 2px solid #ff0000; border-radius: 15px; padding: 20px; margin: 10px 0;">
            <h3 style="color: #ff0000; margin-bottom: 15px;">🔴 Con Strengths</h3>
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
    if st.button("🏠 Back to Landing", key="back_to_landing_history", use_container_width=True):
        st.session_state.current_page = 'landing'
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
    
    if current_page == 'landing':
        render_settings_sidebar()
        render_landing_page()
    elif current_page == 'debate':
        render_debate_arena()
    elif current_page == 'verdict':
        render_verdict_page()
    elif current_page == 'history':
        render_history_page()
    else:
        st.session_state.current_page = 'landing'
        render_settings_sidebar()
        render_landing_page()


if __name__ == "__main__":
    main()
