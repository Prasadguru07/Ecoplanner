# 🌿 EcoPlanner – Sustainable Building Design AI

> **AI-powered energy analysis and eco-material recommendations for sustainable architecture.**
> Built with Python, Streamlit, and Oxlo.ai.

## 🚀 Live Applications

- **[Main Application (Streamlit) →](https://ecoplanner-oxloai.streamlit.app/)** — Full-featured project with all AI analysis capabilities
- **[Live Demo (Netlify) →](https://ecoplanneroxlo.netlify.app/)** — Interactive preview

---

## 📸 Interface Gallery

### 🎨 Hero Image

![EcoPlanner Hero Image](/screenshots/Hero_image.png)

| 🖥️ Main Dashboard & Home | 🤖 AI Analysis & Recommendations |
|:---:|:---:|
| ![Main Dashboard](/screenshots/Home_page.png) | ![AI Recommendations](/screenshots/P1.png) |
| *Overview of project parameters and initial energy modeling.* | *DeepSeek-R1 generated material strategies and carbon offsets.* |

### 📊 Data & Deep Dive

#### **Materials Database & Carbon Intensity**
![Materials Database](/screenshots/P2.png)
*Real-time tracking of embodied carbon vs. thermal performance metrics.*

#### **Material Specifications**
![Materials Table](/screenshots/T1.png)
*The curated 24-material database used for RAG-based AI filtering.*

#### **Final Strategy & Holistic Report**
![Sustainability Strategy](/screenshots/P3.png)
*Final UI output showing the Holistic Sustainability Strategy and export options.*

---

## 🏗️ Architecture

```text
ecoplanner/
├── app.py                  # Streamlit dashboard (UI layer)
├── requirements.txt
├── .env.template           # Copy to .env and fill in API key
├── core/
│   ├── __init__.py
│   ├── energy_sim.py       # ISO 13790 energy model + pybuildingenergy wrapper
│   └── materials_rag.py    # RAG prompt builder + Oxlo.ai API client
└── data/
    └── materials.csv       # 24 eco-friendly materials database

## Quick Start

```bash
# 1. Clone / extract the project
cd ecoplanner

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
cp .env.template .env
# Edit .env → set OXLO_API_KEY=your_key_here

# 4. Launch the dashboard
streamlit run app.py
```

## How It Works

### Energy Model (`core/energy_sim.py`)
- Uses a quasi-steady-state monthly simulation (EN ISO 13790) as the primary engine
- Automatically upgrades to `pybuildingenergy` if the library is installed
- Calculates heating/cooling demand, lighting, EUI, and operational CO₂
- Accounts for: envelope U-values, WWR, internal gains, solar gains, ventilation

### Materials RAG (`core/materials_rag.py`)
- Loads `data/materials.csv` (24 curated eco-materials with full properties)
- Ranks candidates by embodied carbon and filters baselines
- Constructs a structured prompt that **forces** the AI to pick ONLY from the CSV
- Calls Oxlo.ai via the OpenAI-compatible API
- Parses the JSON response and cross-references with the CSV for validation
- Calculates **Carbon Offset** = Baseline Embodied Carbon − Recommended Embodied Carbon

### Carbon Offset Calculation
For each recommended material, the offset is computed against one of four reference baselines:
| Element | Baseline |
|---------|---------|
| Structure | Standard Concrete (410 kg CO₂e/m²) |
| Insulation | Standard Glass Wool (280 kg CO₂e/m²) |
| Masonry | Ordinary Fired Brick (270 kg CO₂e/m²) |
| Timber | Conventional Timber Frame (210 kg CO₂e/m²) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OXLO_API_KEY` | — | Your Oxlo.ai API key (required for AI features) |
| `OXLO_BASE_URL` | `https://api.oxlo.ai/v1` | Oxlo API endpoint |
| `OXLO_MODEL` | `deepseek-r1-8b` | AI model for recommendations (DeepSeek R1 8B) |

## Gmail

- guruprasad2903@gmail.com

## Model used

- deepseek-r1-8b
