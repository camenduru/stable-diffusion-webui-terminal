import os, time
import gradio as gr
from modules import script_callbacks
from subprocess import getoutput

def run_live(command):
  with os.popen(command) as pipe:
    for line in pipe:
      line = line.rstrip()
      print(line)
      yield line

def run_static(command):
    out = getoutput(f"{command}")
    print(out)
    return out

def timeout_test(second):
    start_time = time.time()
    while time.time() - start_time < int(second):
        pass
    msg = "🥳"
    return msg

def clear_out_text():
    return ""

def on_ui_tabs():     
    with gr.Blocks() as terminal:
        with gr.Tab("💻 Terminal"):
            gr.Markdown(
            """
            ```py
            model: wget https://huggingface.co/ckpt/anything-v4.5-vae-swapped/resolve/main/anything-v4.5-vae-swapped.safetensors -O /content/stable-diffusion-webui/models/Stable-diffusion/anything-v4.5-vae-swapped.safetensors
            lora:  wget https://huggingface.co/embed/Sakimi-Chan_LoRA/resolve/main/Sakimi-Chan_LoRA.safetensors -O /content/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/Sakimi-Chan_LoRA.safetensors
            embed: wget https://huggingface.co/embed/bad_prompt/resolve/main/bad_prompt_version2.pt -O /content/stable-diffusion-webui/embeddings/bad_prompt_version2.pt
            vae:   wget https://huggingface.co/ckpt/trinart_characters_19.2m_stable_diffusion_v1/resolve/main/autoencoder_fix_kl-f8-trinart_characters.ckpt -O /content/stable-diffusion-webui/models/VAE/autoencoder_fix_kl-f8-trinart_characters.vae.pt
            zip outputs folder: zip -r /content/outputs.zip /content/stable-diffusion-webui/outputs
            ```
            """)
            with gr.Box():
                terminal_command = gr.Textbox(show_label=False, lines=1, placeholder="command")
                terminal_out_text = gr.Textbox(show_label=False)
                btn_terminl_run_static = gr.Button("run static command")
                btn_terminl_run_static.click(run_static, inputs=terminal_command, outputs=terminal_out_text, show_progress=False)
                btn_terminl_run_live = gr.Button("run live command")
                btn_terminl_run_live.click(run_live, inputs=terminal_command, outputs=terminal_out_text, show_progress=False) 
        with gr.Tab("Tests"):
            with gr.Box():
                test_command = gr.Textbox(show_label=False, lines=1, placeholder="command")
                test_out_text = gr.Textbox(show_label=False)
                btn_timeout_test = gr.Button("timeout test")
                btn_timeout_test.click(timeout_test, inputs=test_command, outputs=test_out_text, show_progress=False)
                btn_clear = gr.Button("clear")
                btn_clear.click(clear_out_text, inputs=[], outputs=test_out_text, show_progress=False)
    return (terminal, "Terminal", "terminal"),
script_callbacks.on_ui_tabs(on_ui_tabs)
