# DebateBot: Two Sides and a Judge ⚔️

An AI-powered debate arena where two AI agents battle using arguments while a Judge evaluates and declares a winner. Built with Streamlit, LangGraph, Groq API, and Tavily Search API.

## Features

### 🎮 Fighting Game-Inspired Interface
- **Health Bars**: Real-time persuasion bars for Pro (green) and Con (red) agents
- **Round Tracking**: Visual progress indicators showing current round and debate progress
- **Sequential Display**: Messages appear one at a time like WhatsApp conversations
- **Typing Animations**: Thinking animations while agents generate responses
- **Scoreboard**: Live scoring with current leader display

### 🏠 Landing Page
- Topic input with validation
- Suggested debate topics
- Random topic generator
- Comprehensive settings sidebar

### ⚙️ Customizable Settings
- **Number of Rounds**: 3, 5, or 7 rounds
- **Argument Length**: Short, medium, or long
- **Judge Personality**: Balanced, strict, or lenient
- **Citation Toggle**: Show/hide citations during debate

### ⚔️ Live Debate Arena
- Real-time health bar updates
- Turn-by-turn message display
- Progress indicators
- Round-wise scoreboard
- Citation display with Tavily search results
- Auto-play functionality
- "Judge is observing" status during debate

### ⚖️ Judge Verdict Screen
- Winner announcement with visual effects
- Final score display
- Best argument and rebuttal highlights
- Key strengths of both sides
- Detailed final reasoning
- Premium result card design

### 📜 Debate History
- Save debates with unique IDs
- Search debates by topic
- View full debate details
- Delete individual debates
- Statistics dashboard

### 📥 PDF Export
- Export debates to PDF
- Include topic, transcript, citations, and verdict
- Professional formatting

### 🎨 Theme System
- Dark theme with fighting game aesthetics
- Green for Pro, Red for Con, Gold for Judge
- Custom CSS animations and transitions
- Responsive design

## Live Demo:  https://debatebot-langgraph.streamlit.app

## Architecture

The project uses a single-file architecture (`app.py`) with:

- **LangGraph**: Agent orchestration for structured debate flow
- **Groq API**: LLM provider for Pro, Con, and Judge agents
- **Tavily API**: Evidence tool for real citations and sources
- **Streamlit**: Frontend, backend, state management, and session handling
- **ReportLab**: PDF generation

```
DebateBot_LangGraph/
├── app.py                 # Main application (single file)
├── requirements.txt       # Python dependencies
├── .env.example          # API key template
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── history/              # Saved debate data (JSON)
└── exports/              # Exported PDF files
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Groq API key
- Tavily API key

### Step 1: Navigate to Project Directory
```bash
cd C:\Users\asus\CascadeProjects\DebateBot_LangGraph
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

Edit `.env` and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Get API Keys:**
- **Groq API**: Sign up at [console.groq.com](https://console.groq.com)
- **Tavily API**: Sign up at [tavily.com](https://tavily.com)

### Step 5: Run the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Usage

### Starting a New Debate

1. **Enter a Topic**: Type your debate topic or select from suggested topics
2. **Configure Settings**: Adjust rounds, argument length, and judge personality in the sidebar
3. **Start Debate**: Click the "START DEBATE" button
4. **Watch the Battle**: Click "Next Round" to advance through the debate
5. **Call the Judge**: After all rounds complete, click "CALL JUDGE" to see the verdict
6. **View Transcript**: Click "Show Debate Transcript" to see the full conversation

### Managing Debate History

1. **Access History**: Click "View Debate History" from any page
2. **Search**: Use the search bar to find specific debates
3. **View Details**: Click "View" on any debate card to see full details
4. **Delete**: Click "Delete" to remove a debate

### Exporting Debates

1. **From Verdict Page**: After a debate completes, click "Export PDF"
2. **Download**: The PDF file will be downloaded automatically

## How It Works

### Agent System

The debate uses three AI agents powered by Groq's Llama 3.3 70B model:

1. **Pro Agent**: Argues in favor of the topic
2. **Con Agent**: Argues against the topic
3. **Judge Agent**: Evaluates arguments and declares a winner

### Debate Flow

1. User selects topic and settings
2. Pro Agent generates opening argument with evidence
3. Con Agent generates rebuttal with counter-evidence
4. Process repeats for each round
5. Judge analyzes all arguments and provides verdict
6. Results displayed with detailed analysis

### Evidence System

- Uses Tavily Search API to find real sources
- Agents cite sources in their arguments
- Citations displayed with source links
- Supports fact-based debates

### State Management

- Streamlit `session_state` manages all application state
- Debate progress tracked in real-time
- Health/persuasion scores updated each round
- History persisted to JSON files

## Configuration

### Environment Variables

Required variables in `.env`:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Streamlit Configuration

The `.streamlit/config.toml` file includes:
- Dark theme with gold accents
- Server settings
- Browser configuration

## Customization

### Modifying Agent Prompts

Edit the system prompts in `app.py`:
- `create_pro_agent()` - Pro agent behavior
- `create_con_agent()` - Con agent behavior
- `create_judge_agent()` - Judge evaluation criteria

### Changing LLM Model

Modify the `model` parameter in agent creation functions:
```python
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Change this
    temperature=0.7,
    api_key=GROQ_API_KEY
)
```

Available Groq models:
- `llama-3.3-70b-versatile`
- `mixtral-8x7b-32768`
- `gemma-7b-it`

### Adjusting Scoring Logic

Modify the score updates in `simulate_debate_round()`:
```python
# Current: Random score changes
pro_change = random.uniform(-5, 10)
con_change = random.uniform(-10, 5)

# Replace with actual evaluation logic
```

## Troubleshooting

### Issue: API Key Errors
**Solution**: Ensure `.env` file exists with valid API keys

### Issue: Application won't start
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Issue: History not saving
**Solution**: Ensure the `history/` directory has write permissions

### Issue: PDF export fails
**Solution**: Ensure the `exports/` directory exists and is writable

### Issue: Slow response times
**Solution**: Groq API may have rate limits; consider reducing rounds or argument length

## API Integration Details

### Groq API Integration

The application uses LangChain's `ChatGroq` integration:
- Model: Llama 3.3 70B Versatile
- Temperature: 0.7 for agents, 0.3 for judge
- Context: Previous messages included for coherence

### Tavily API Integration

The application uses LangChain's `TavilySearchResults` tool:
- Max results: 3 per search
- Search depth: Advanced
- Includes: Answer, raw content, URLs

## Deployment

### Local Deployment

```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. **Push to GitHub**: Upload your code to a GitHub repository
2. **Add Secrets**: Add GROQ_API_KEY and TAVILY_API_KEY as secrets
3. **Deploy**: Go to [share.streamlit.io](https://share.streamlit.io) and deploy

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t debatebot .
docker run -p 8501:8501 debatebot
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with Streamlit
- AI agents powered by Groq (Llama 3.3)
- Evidence retrieval via Tavily Search
- Agent orchestration with LangGraph
- Inspired by fighting games and esports

## Contact

For questions or support, please open an issue on the GitHub repository.

---

**DebateBot: Where AI Minds Clash in Intellectual Combat** ⚔️
