# multi-agent-system/agents/test_agents.py
"""
Quick test to run CT-Agent, Audio-Agent, Metadata-Agent, Fusion-Agent, Report-Agent sequentially.
Update paths below to real CT/audio files before running.
"""

from multi_agent_system.observability.logging_config import configure_logging
configure_logging()

from multi_agent_system.agents.ct_agent import CTAgent
from multi_agent_system.agents.audio_agent import AudioAgent
from multi_agent_system.agents.metadata_agent import MetadataAgent
from multi_agent_system.agents.fusion_agent import FusionAgent
from multi_agent_system.agents.report_agent import ReportAgent

# --- configure agents ---
ct_agent = CTAgent(device="cpu", unet_weights=None, classifier_weights=None)
audio_agent = AudioAgent(device="cpu", weight_path=None)
meta_agent = MetadataAgent(device="cpu", weight_path=None)
fusion_agent = FusionAgent(device="cpu", fusion_weights=None, metadata_weights=None)
report_agent = ReportAgent(api_key_env_var="GEMINI_API_KEY")

# --- update these to actual test files on your machine ---
CT_PATH = "sample_ct_folder"   # replace with DICOM folder or .nii file
AUDIO_PATH = "example.wav"     # replace with a real wav file

# example metadata
metadata = {
    "age": 60,
    "sex": "male",
    "smoking_history": 2,
    "symptoms": ["cough", "short_breath"]
}

def main():
    print("Running CT Agent ...")
    ct_out = ct_agent.run(CT_PATH, is_dicom=True)
    print("CT score:", ct_out["ct_score"])

    print("Running Audio Agent ...")
    audio_out = audio_agent.run(AUDIO_PATH)
    print("Audio score:", audio_out["audio_score"])

    print("Running Metadata Agent ...")
    meta_out = meta_agent.run(metadata)
    print("Metadata embedding shape:", meta_out["metadata_embedding"].shape)

    print("Running Fusion Agent ...")
    fusion_out = fusion_agent.run(ct_out["ct_embedding"], audio_out["audio_embedding"], metadata)
    print("Fusion score:", fusion_out["fusion_score"])

    print("Generating Report ...")
    ct_summary = f"CT score: {ct_out['ct_score']:.3f}"
    audio_summary = f"Audio score: {audio_out['audio_score']:.3f}"
    metadata_summary = str(metadata)
    report = report_agent.run(ct_summary, audio_summary, metadata_summary, fusion_out["fusion_score"])
    print("Report (first 500 chars):")
    print(report["report_text"][:500])

if __name__ == "__main__":
    main()
