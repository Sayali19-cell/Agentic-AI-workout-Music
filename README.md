# Agentic-AI-workout-Music
Agentic AI Workout Music system a Streamlit app + agent pipeline that synthesizes short workout music segments based on simulated sensor input and user preferences. The project demonstrates how to wire controller agents (AutoGen or local rule-based) to a music generator (Meta MusicGen or a local synthesizer) and play output in a browser UI. 



Agentic AI Workout Music — README

Welcome — this repository contains a prototype Agentic AI Workout Music system: a Streamlit app + agent pipeline that generates or synthesizes short workout music segments based on simulated sensor input (intensity / heart-rate) and user preferences. The project demonstrates how to wire controller agents (AutoGen or local rule-based) to a music generator (Meta MusicGen or a local synthesizer) and play output in a browser UI.

This README covers: what the project is, how it works, setup & install, running locally (Windows / macOS / Linux), environment variables, fallbacks and troubleshooting, and deployment / next steps.



Project overview:

Agentic AI Workout Music is a prototype that demonstrates:

a controller agent that decides music style & tempo from sensor input (intensity, user prefs),

a music generator that produces short audio segments (via MusicGen if available or a local synthesizer fallback),

a web UI (Streamlit) where you can simulate intensity and generate music,

orchestrator patterns implemented two ways: (a) LangGraph (optional) or (b) small built-in pipeline (no external orchestrator) for portability.

This is a research / prototype repo — built for exploration and testing, not production.

Features

Controller agent options:

AutoGen (OpenAI): use AutoGen AgentChat if OPENAI_API_KEY is set (recommended for richer plans).

Local rule-based fallback: deterministic mapping from intensity → BPM/style/prompt.

Music generation options:

MusicGen (transformers facebook/musicgen-small) if transformers + torch available.

Synthesized fallback (sine + percussive loop) when MusicGen not available.

Optional orchestration:

LangGraph pipeline if you pip install langgraph. If not present, a lightweight pipeline runs instead.

Streamlit UI with intensity slider, user profile (genre) settings, and audio playback.

Graceful fallbacks and informative log messages (Streamlit shows what the app is doing).

Architecture
[Streamlit UI] -> (intensity, profile)
      |
  ┌───v─────────────────────────┐
  │ Simple Orchestration Layer │  (LangGraph optional)
  └───┬────────────────────────┘
      |
  [Sensor Node] -> [Controller Node (AutoGen or local)] -> [Music Node (MusicGen or synth)] -> [Player Node]


Controller outputs a plan JSON: { phase, bpm, music_style, energy, prompt }. Music node uses prompt to generate a short audio snippet and returns a WAV file path to the player.

Files in this repo

agentic_autogen_langgraph.py — main Streamlit prototype that integrates AutoGen + LangGraph (optional) and MusicGen/fallback synth.

agentic_workout.py — earlier single-file prototype (if included).

requirements.txt — suggested packages for development (see below).

README.md — this file.

Note: if a file name differs in your local copy (e.g., agentic_workout.py) adapt the run commands accordingly.
