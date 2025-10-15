"""
agentic_autogen_langgraph.py

Run:
    pip install -r requirements.txt
    streamlit run agentic_autogen_langgraph.py

requirements.txt (suggested):
    streamlit
    langgraph --pre
    autogen-agentchat
    autogen-ext[openai]
    transformers>=4.30
    torch
    accelerate
    scipy
    soundfile
    numpy

Notes:
- If you don't have an LLM key (OPENAI_API_KEY or other supported by AutoGen ext),
  the controller will use a local fallback.
- MusicGen (transformers) is optional. If missing, the app synthesizes a short beep-loop.
"""
import os
import asyncio
import tempfile
import time
import json
from pathlib import Path

import streamlit as st

# LangGraph imports
try:
    from langgraph.graph import StateGraph, MessagesState, START, END
    LANGGRAPH_AVAILABLE = True
except Exception as e:
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_ERROR = str(e)

# AutoGen imports (optional)
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except Exception as e:
    AUTOGEN_AVAILABLE = False
    AUTOGEN_ERROR = str(e)

# MusicGen / transformers imports (optional)
try:
    from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
    import torch
    import scipy.io.wavfile as wavfile
    import numpy as np
    MUSICGEN_AVAILABLE = True
except Exception as e:
    MUSICGEN_AVAILABLE = False
    MUSICGEN_ERROR = str(e)
    import numpy as np
    import scipy.io.wavfile as wavfile

# --------------------------
# Helper: Local controller fallback
# --------------------------
def local_controller(intensity: float, profile: dict) -> dict:
    """Deterministic mapping from intensity->music plan."""
    if intensity < 0.25:
        phase = "warmup"; bpm = 90; energy = "low"
        style = "ambient chill instrumental with soft pads and light percussion"
    elif intensity < 0.6:
        phase = "cardio"; bpm = 120; energy = "medium"
        style = "upbeat pop-electronic with steady drums"
    else:
        phase = "hiit"; bpm = 140; energy = "high"
        style = "high-energy electronic dance, driving bass, aggressive beat"

    pref = profile.get("genre","")
    style_text = f"{pref} style, {style}" if pref else style
    prompt = (
        f"Create a {phase} workout music loop. Style: {style_text}. Tempo: ~{bpm} BPM. "
        "Duration: 20 seconds. No vocals. Clear beats suitable for workout."
    )
    return {"phase": phase, "bpm": bpm, "music_style": style_text, "energy": energy, "prompt": prompt}

# --------------------------
# Helper: AutoGen-based controller (async wrapper)
# --------------------------
async def _run_autogen_controller_async(intensity: float, profile: dict) -> dict:
    """
    Use AutoGen AgentChat AssistantAgent with OpenAI client (if available).
    Falls back to local_controller if not configured.
    """
    # if AutoGen not installed, raise -> outer wrapper will fallback
    if not AUTOGEN_AVAILABLE:
        raise RuntimeError(f"AutoGen not available: {AUTOGEN_ERROR}")

    # require OPENAI_API_KEY (or another configured provider supported by autogen-ext)
    api_key = os.getenv("OPENAI_API_KEY", None)
    if not api_key:
        # try to see if other environment config exists for autogen_ext - but we keep simple
        raise RuntimeError("OPENAI_API_KEY not set. Set it to use AutoGen controller.")

    # create model client and agent
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")  # adjust to correct model name
    agent = AssistantAgent("controller_agent", model_client=model_client,
                           system_message="You are a controller that returns JSON plan for music generation.")
    try:
        task = f"Given intensity={round(float(intensity),2)} and profile={json.dumps(profile)}, return a JSON with keys: phase,bpm,music_style,energy,prompt. Respond with JSON only."
        # run agent using asyncio
        result_text = await agent.run(task=task)
        # agent.run often returns a string — parse JSON
        import re
        m = re.search(r"\{.*\}", result_text, re.S)
        if m:
            plan = json.loads(m.group(0))
        else:
            plan = json.loads(result_text)
        await model_client.close()
        return plan
    finally:
        try:
            await model_client.close()
        except Exception:
            pass

def autogen_controller(intensity: float, profile: dict) -> dict:
    """Sync wrapper around the async AutoGen agent call. If anything fails, raise."""
    return asyncio.run(_run_autogen_controller_async(intensity, profile))

# --------------------------
# Helper: Music generation
# --------------------------
def generate_music_with_musicgen(prompt: str, out_path: str, max_new_tokens: int = 1024):
    """
    Use MusicGen (transformers) if available. Returns the output path.
    """
    global _MUSICGEN_MODEL, _MUSICGEN_PROCESSOR
    if not MUSICGEN_AVAILABLE:
        raise RuntimeError(f"MusicGen not available: {MUSICGEN_ERROR}")

    if "_MUSICGEN_MODEL" not in globals():
        _MUSICGEN_PROCESSOR = MusicgenProcessor.from_pretrained("facebook/musicgen-small")
        _MUSICGEN_MODEL = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        if torch.cuda.is_available():
            _MUSICGEN_MODEL = _MUSICGEN_MODEL.to("cuda")
    inputs = _MUSICGEN_PROCESSOR(text=[prompt], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    audio_waves = _MUSICGEN_MODEL.generate(**inputs, max_new_tokens=max_new_tokens)
    audio_np = audio_waves[0].cpu().numpy()
    sample_rate = 32000
    if audio_np.ndim == 2:
        audio_mono = audio_np.mean(axis=0)
    else:
        audio_mono = audio_np
    audio_mono = audio_mono / max(1e-9, abs(audio_mono).max())
    audio_int16 = (audio_mono * 32767).astype('int16')
    wavfile.write(out_path, sample_rate, audio_int16)
    return out_path

# --------------------------
# Fallback: simple synthesized loop (sine-bass + beat) if MusicGen missing
# --------------------------
def synthesize_loop_bpm_bass(bpm:int, duration_sec:int, out_path:str, sample_rate=22050):
    """Synthesize a short percussive loop + sub-bass tone to simulate music."""
    t = np.linspace(0, duration_sec, int(sample_rate*duration_sec), endpoint=False)
    # simple sub-bass sine
    bass = 0.25 * np.sin(2*np.pi*55*t) * np.exp(-0.0005*np.arange(t.size))
    # create kick-like percussive hits at each beat
    beat_interval = 60.0 / bpm
    audio = np.zeros_like(t)
    for i in range(int(duration_sec/beat_interval)+2):
        center = int(i * beat_interval * sample_rate)
        # short percussive click
        if center < t.size:
            length = int(0.05*sample_rate)
            env = np.linspace(1.0, 0.0, length)
            audio[center:center+length] += env * (np.random.randn(length) * 0.5 + 1.0)
    mix = audio + bass
    # normalize
    mix = mix / max(1e-9, abs(mix).max())
    audio_int16 = (mix * 32767).astype('int16')
    wavfile.write(out_path, sample_rate, audio_int16)
    return out_path

# --------------------------
# LangGraph nodes
# --------------------------
def sensor_node(state: dict):
    """Node that reads the provided invocation state (intensity/profile) and forwards it."""
    # state expected to contain 'intensity' and 'profile' keys passed in invocation
    intensity = state.get("intensity", 0.5)
    profile = state.get("profile", {})
    return {"intensity": float(intensity), "profile": profile}

# def controller_node(state: dict):
#     """
#     Node to produce a plan: try AutoGen agent if available and configured; otherwise local fallback.
#     Returns: plan dict
#     """
#     intensity = state["intensity"]
#     profile = state["profile"]
#     # Try AutoGen if available and keys present
#     try:
#         if AUTOGEN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
#             st.write("Controller: using AutoGen agent.")
#             plan = autogen_controller(intensity, profile)
#             st.write("AutoGen returned plan:", plan)
#             return {"plan": plan}
#         else:
#             st.write("Controller: using local fallback.")
#             plan = local_controller(intensity, profile)
#             st.write("Local plan:", plan)
#             return {"plan": plan}
#     except Exception as e:
#         st.write("Controller exception, falling back locally:", e)
#         plan = local_controller(intensity, profile)
#         return {"plan": plan}

def controller_node(state: dict):
    """
    Node to produce a plan: try AutoGen agent (OpenAI via autogen-ext) if available and configured;
    otherwise use the local deterministic fallback.
    """
    intensity = state.get("intensity", 0.5)
    profile = state.get("profile", {})

    # Prefer AutoGen if available and OPENAI_API_KEY is set
    try:
        if AUTOGEN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            st.write("Controller: using AutoGen agent (OpenAI).")
            try:
                plan = autogen_controller(intensity, profile)
                # basic validation
                if not isinstance(plan, dict) or "prompt" not in plan:
                    st.write("AutoGen returned unexpected format; falling back to local controller. Output:", plan)
                    plan = local_controller(intensity, profile)
                else:
                    st.write("AutoGen returned plan:", plan)
                return {"plan": plan}
            except Exception as e:
                st.write("AutoGen controller error:", e)
                st.write("Falling back to local_controller.")
                plan = local_controller(intensity, profile)
                return {"plan": plan}
        else:
            # AutoGen not available or key not set — use local fallback
            if not AUTOGEN_AVAILABLE:
                st.write("AutoGen not installed or import failed:", AUTOGEN_ERROR if 'AUTOGEN_ERROR' in globals() else "unknown")
            else:
                st.write("OPENAI_API_KEY not found; not using AutoGen.")
            plan = local_controller(intensity, profile)
            st.write("Local plan:", plan)
            return {"plan": plan}
    except Exception as e:
        st.write("Unexpected controller_node exception; using local fallback:", e)
        plan = local_controller(intensity, profile)
        return {"plan": plan}


def music_node(state: dict):
    """Node to produce audio for state['plan'] -> returns {'wav_path': path}"""
    plan = state["plan"]
    prompt = plan.get("prompt", "")
    bpm = plan.get("bpm", 120)
    duration = 20
    tmp = Path(tempfile.gettempdir()) / f"agentic_music_{int(time.time())}.wav"
    try:
        if MUSICGEN_AVAILABLE:
            st.write("Music node: generating via MusicGen...")
            out = generate_music_with_musicgen(prompt, str(tmp))
            st.write("MusicGen produced:", out)
            return {"wav_path": out}
        else:
            st.write("Music node: synthesizing fallback audio...")
            out = synthesize_loop_bass_bpm = synthesize_loop_bpm_bass(int(bpm), duration, str(tmp))
            st.write("Synth produced:", out)
            return {"wav_path": out}
    except Exception as e:
        st.write("Music node failed; synthesizing fallback:", e)
        out = synthesize_loop_bpm_bass(int(bpm), duration, str(tmp))
        return {"wav_path": out}

def player_node(state: dict):
    """Return wav path (state expected to have 'wav_path')"""
    return {"wav_path": state.get("wav_path")}

# --------------------------
# Build LangGraph graph (functional, compact)
# --------------------------
def build_graph():
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(f"LangGraph not available: {LANGGRAPH_ERROR}")
    graph = StateGraph(MessagesState)
    graph.add_node(sensor_node, name="sensor_node")
    graph.add_node(controller_node, name="controller_node")
    graph.add_node(music_node, name="music_node")
    graph.add_node(player_node, name="player_node")
    graph.add_edge(START, "sensor_node")
    graph.add_edge("sensor_node", "controller_node")
    graph.add_edge("controller_node", "music_node")
    graph.add_edge("music_node", "player_node")
    graph.add_edge("player_node", END)
    compiled = graph.compile()
    return compiled

# --------------------------
# Streamlit UI + orchestration invocation
# --------------------------
st.set_page_config(page_title="Agentic Workout (AutoGen + LangGraph)", layout="centered")
st.title("🎧 Agentic Workout — AutoGen + LangGraph Prototype")

st.markdown("LangGraph: " + ("OK" if LANGGRAPH_AVAILABLE else f"NOT available: {LANGGRAPH_ERROR[:140]}"))
st.markdown("AutoGen: " + ("OK" if AUTOGEN_AVAILABLE else f"NOT available: {AUTOGEN_ERROR[:140]}"))
st.markdown("MusicGen: " + ("OK" if MUSICGEN_AVAILABLE else f"NOT available: {MUSICGEN_ERROR[:140]}"))

st.sidebar.header("User prefs")
genre = st.sidebar.selectbox("Preferred genre", ["electronic","pop","rock","hip hop","instrumental"])
profile = {"genre": genre}

st.sidebar.markdown("If you want AutoGen controller, set OPENAI_API_KEY in your environment (or configure other AutoGen extensions).")

intensity = st.slider("Intensity (simulate HR/pace)", 0.0, 1.0, 0.5, 0.01)
generate = st.button("Run agentic pipeline")

if "graph" not in st.session_state:
    try:
        st.session_state.graph = build_graph()
    except Exception as e:
        st.error("Could not build LangGraph: " + str(e))
        st.stop()

if generate:
    st.info("Invoking LangGraph workflow...")
    # graph expects a "messages"-style state for MessagesState; we pass a dict with keys used by sensor_node
    invoke_state = {"intensity": float(intensity), "profile": profile, "messages": [{"role":"user","content":"start"}]}
    try:
        result = st.session_state.graph.invoke(invoke_state)
        # result is a state dict; get wav path
        # depending on LangGraph version, structure may vary; defensively search
        wav_path = None
        if isinstance(result, dict):
            # search nested for wav_path
            def find_wav(d):
                if not isinstance(d, dict):
                    return None
                if "wav_path" in d:
                    return d["wav_path"]
                for v in d.values():
                    res = find_wav(v)
                    if res:
                        return res
                return None
            wav_path = find_wav(result)
        if wav_path and Path(wav_path).exists():
            st.success("Audio ready:")
            with open(wav_path,"rb") as f:
                st.audio(f.read(), format="audio/wav")
        else:
            st.warning("No audio produced; graph result: " + str(result)[:1000])
    except Exception as e:
        st.error("Graph invocation failed: " + str(e))
        # fallback single-run: run controller_node + music_node manually
        st.info("Falling back to direct controller -> music generation.")
        plan = None
        try:
            if AUTOGEN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                plan = autogen_controller(float(intensity), profile)
            else:
                plan = local_controller(float(intensity), profile)
        except Exception as ee:
            st.write("Controller fallback error:", ee)
            plan = local_controller(float(intensity), profile)
        st.write("Plan:", plan)
        tmp = Path(tempfile.gettempdir()) / f"agentic_music_fallback_{int(time.time())}.wav"
        try:
            if MUSICGEN_AVAILABLE:
                out = generate_music_with_musicgen(plan["prompt"], str(tmp))
            else:
                out = synthesize_loop_bpm_bass(int(plan.get("bpm",120)), 20, str(tmp))
            with open(out,"rb") as f:
                st.audio(f.read(), format="audio/wav")
        except Exception as e2:
            st.error("Music generation fallback failed: " + str(e2))
