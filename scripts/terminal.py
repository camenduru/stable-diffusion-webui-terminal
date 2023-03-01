import os, time
import gradio as gr
from modules import script_callbacks
from subprocess import getoutput
import launch

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

def install_diffusers():
    if not launch.is_installed("diffusers"):
        launch.run_pip("install diffusers==0.13.1", "diffusers==0.13.1 requirements for diffusers extension")
        yield "diffusers==0.13.1 requirements for diffusers extension"
    if not launch.is_installed("transformers"):
        launch.run_pip("install transformers==4.26.1", "transformers==4.26.1 requirements for diffusers extension")
        yield "transformers==4.26.1 requirements for diffusers extension"
    if not launch.is_installed("ftfy"):
        launch.run_pip("install ftfy==6.1.1", "ftfy==6.1.1 requirements for diffusers extension")
        yield "ftfy==6.1.1 requirements for diffusers extension"
    if not launch.is_installed("accelerate"):
        launch.run_pip("install accelerate==0.16.0", "accelerate==0.16.0 requirements for diffusers extension")
        yield "accelerate==0.16.0 requirements for diffusers extension"
    if not launch.is_installed("bitsandbytes"):
        launch.run_pip("install bitsandbytes==0.37.0", "bitsandbytes==0.37.0 requirements for diffusers extension")
        yield "bitsandbytes==0.37.0 requirements for diffusers extension"
    if not launch.is_installed("safetensors"):
        launch.run_pip("install safetensors==0.2.8", "safetensors==0.2.8 requirements for diffusers extension")
        yield "safetensors==0.2.8 requirements for diffusers extension"

def on_ui_tabs():     
    with gr.Blocks() as terminal:
        with gr.Tab("Terminal"):
            gr.Markdown(
            """
            ### 💻 Terminal
            ```py
            model: wget https://huggingface.co/ckpt/anything-v4.5-vae-swapped/resolve/main/anything-v4.5-vae-swapped.safetensors -O /content/stable-diffusion-webui/models/Stable-diffusion/anything-v4.5-vae-swapped.safetensors
            lora:  wget https://huggingface.co/embed/Sakimi-Chan_LoRA/resolve/main/Sakimi-Chan_LoRA.safetensors -O /content/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/Sakimi-Chan_LoRA.safetensors
            embed: wget https://huggingface.co/embed/bad_prompt/resolve/main/bad_prompt_version2.pt -O /content/stable-diffusion-webui/embeddings/bad_prompt_version2.pt
            vae:   wget https://huggingface.co/ckpt/trinart_characters_19.2m_stable_diffusion_v1/resolve/main/autoencoder_fix_kl-f8-trinart_characters.ckpt -O /content/stable-diffusion-webui/models/VAE/autoencoder_fix_kl-f8-trinart_characters.vae.pt
            zip outputs folder: zip -r /content/outputs.zip /content/stable-diffusion-webui/outputs
            ```
            """)
            with gr.Group():
                command = gr.Textbox(show_label=False, max_lines=5, placeholder="command")
                out_text = gr.Textbox(show_label=False)
                btn_static = gr.Button("run static command")
                btn_static.click(run_static, inputs=command, outputs=out_text)
        with gr.Tab("Training"):
            with gr.Tab("Install Diffusers"):
                with gr.Group():
                    out_text = gr.Textbox(show_label=False)
                    btn_install_diffusers = gr.Button("Install Diffusers")
                    btn_install_diffusers.click(install_diffusers, [], outputs=out_text)
            with gr.Tab("Train Dreambooth"):
                with gr.Row():
                    with gr.Column:
                        gr.Markdown(
                        """
                        ```py
                        rm -rf /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/dreambooth/output_dir/*\n
                        pip install -U diffusers==0.13.1 transformers==4.26.1 ftfy==6.1.1 accelerate==0.16.0 bitsandbytes==0.37.0 safetensors==0.2.8\n
                        python /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/dreambooth/train_dreambooth.py \\
                            --pretrained_model_name_or_path="JosephusCheung/ACertainty"  \\
                            --instance_data_dir="/content/drive/MyDrive/AI/training/parkminyoung" \\
                            --output_dir="/content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/dreambooth/output_dir" \\
                            --learning_rate=5e-6 \\
                            --max_train_steps=650 \\
                            --instance_prompt="parkminyoung" \\
                            --resolution=512 \\
                            --center_crop \\
                            --train_batch_size=1 \\
                            --gradient_accumulation_steps=1 \\
                            --max_grad_norm=1.0 \\
                            --mixed_precision="fp16" \\
                            --gradient_checkpointing \\
                            --enable_xformers_memory_efficient_attention \\
                            --use_8bit_adam\n
                            --with_prior_preservation \\
                            --class_data_dir="/content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/dreambooth/class_data_dir" \\
                            --prior_loss_weight=1.0 \\
                            --sample_batch_size=2 \\
                            --class_prompt="person" \\
                            --seed=69 \\
                            --num_class_images=12 \\ \n
                        python /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/dreambooth/convert_diffusers_to_original_stable_diffusion.py --model_path /content/stable-diffusion-webui/extensions/stable-diffusion-webui-diffusers/output_dir --checkpoint_path /content/stable-diffusion-webui/models/Stable-diffusion/parkminyoung.ckpt
                        ```
                        """)
                    with gr.Column():
                        with gr.Group():
                            command = gr.Textbox(show_label=False, lines=5, placeholder="command")
                            out_text = gr.Textbox(show_label=False, lines=5)
                            btn_run_live = gr.Button("run live command")
                            btn_run_live.click(run_live, inputs=command, outputs=out_text)
            with gr.Tab("Train LoRA"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                        """
                        ```py
                        rm -rf /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/lora/output_dir/*\n
                        pip install -U diffusers==0.13.1 transformers==4.26.1 ftfy==6.1.1 accelerate==0.16.0 bitsandbytes==0.37.0 safetensors==0.2.8\n
                        python /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/lora/train_dreambooth_lora.py \\
                            --pretrained_model_name_or_path="JosephusCheung/ACertainty"  \\
                            --instance_data_dir="/content/drive/MyDrive/AI/training/parkminyoung" \\
                            --output_dir="/content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/lora/output_dir" \\
                            --learning_rate=5e-6 \\
                            --max_train_steps=650 \\
                            --instance_prompt="parkminyoung" \\
                            --resolution=512 \\
                            --center_crop \\
                            --train_batch_size=1 \\
                            --gradient_accumulation_steps=1 \\
                            --max_grad_norm=1.0 \\
                            --mixed_precision="fp16" \\
                            --gradient_checkpointing \\
                            --enable_xformers_memory_efficient_attention \\
                            --use_8bit_adam\n
                            --with_prior_preservation \\
                            --class_data_dir="/content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/lora/class_data_dir" \\
                            --prior_loss_weight=1.0 \\
                            --sample_batch_size=2 \\
                            --class_prompt="person" \\
                            --seed=69 \\
                            --num_class_images=12 \\ \n
                        python /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/lora/convert_diffusers_to_original_stable_diffusion_lora.py
                        cp /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/lora/output_dir/pytorch_lora_weights.safetensors /content/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/parkminyoung.safetensors
                        ```
                        """)
                    with gr.Column():
                        with gr.Group():
                            command = gr.Textbox(show_label=False, lines=5, placeholder="command")
                            out_text = gr.Textbox(show_label=False, lines=5)
                            btn_run_live = gr.Button("run live command")
                            btn_run_live.click(run_live, inputs=command, outputs=out_text)
        with gr.Tab("Tests"):
            with gr.Group():
                command = gr.Textbox(show_label=False, lines=1, placeholder="command")
                out_text = gr.Textbox(show_label=False)
                btn_timeout_test = gr.Button("timeout test")
                btn_timeout_test.click(timeout_test, inputs=command, outputs=out_text)
    return (terminal, "Terminal", "terminal"),
script_callbacks.on_ui_tabs(on_ui_tabs)
