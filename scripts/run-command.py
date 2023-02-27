import os
import gradio as gr
from modules import scripts, script_callbacks

def run(command):
  with os.popen(command) as pipe:
    for line in pipe:
      line = line.rstrip()
      print(line)
      yield line

def on_ui_tabs():     
    with gr.Blocks() as run-command:
        gr.Markdown(
        """
        ### 🦒 Colab Run Command
        ```py
        model: wget https://huggingface.co/ckpt/anything-v4.5-vae-swapped/resolve/main/anything-v4.5-vae-swapped.safetensors -O /content/stable-diffusion-webui/models/Stable-diffusion/anything-v4.5-vae-swapped.safetensors
        lora:  wget https://huggingface.co/embed/Sakimi-Chan_LoRA/resolve/main/Sakimi-Chan_LoRA.safetensors -O /content/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/Sakimi-Chan_LoRA.safetensors
        embed: wget https://huggingface.co/embed/bad_prompt/resolve/main/bad_prompt_version2.pt -O /content/stable-diffusion-webui/embeddings/bad_prompt_version2.pt
        vae:   wget https://huggingface.co/ckpt/trinart_characters_19.2m_stable_diffusion_v1/resolve/main/autoencoder_fix_kl-f8-trinart_characters.ckpt -O /content/stable-diffusion-webui/models/VAE/autoencoder_fix_kl-f8-trinart_characters.vae.pt
        zip outputs folder: zip -r /content/outputs.zip /content/stable-diffusion-webui/outputs
        ```
        """)
        with gr.Group():
            with gr.Box():
                command = gr.Textbox(show_label=False, max_lines=1, placeholder="command")
                out_text = gr.Textbox(show_label=False)
                btn_run = gr.Button("run command")
                btn_run.click(run, inputs=command, outputs=out_text)
    return (run-command, "Run Command", "run-command"),
script_callbacks.on_ui_tabs(on_ui_tabs)
