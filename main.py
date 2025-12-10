from backend.backend import Backend
from ui.gradio_interface import launch_ui

if __name__ == "__main__":
    be = Backend(
        model_root="./models",
        llm_subdir="deepseek-finetuned-go",
        valuenet_path="./valuenet.h5",
        #device="cuda",          # 无GPU可改 "cpu"
        device="cpu",
        prior_mix=0.7,          # 先验融合：LLM权重
        temperature=1.0
    )
    launch_ui(backend=be, server_name="127.0.0.1", server_port=7860, share=False)
