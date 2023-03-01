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
                command = gr.Textbox(show_label=False, lines=1, placeholder="command")
                out_text = gr.Textbox(show_label=False)
                btn_static = gr.Button("run static command")
                btn_static.click(run_static, inputs=command, outputs=out_text, show_progress=False)
        with gr.Tab("Install"):
            with gr.Box():
                command = gr.Textbox(show_label=False, lines=1, value="pip install -U diffusers==0.13.1 transformers==4.26.1 ftfy==6.1.1 accelerate==0.16.0 bitsandbytes==0.37.0 safetensors==0.2.8")
                out_text = gr.Textbox(show_label=False)
                btn_run_live = gr.Button("Install Diffusers")
                btn_run_live.click(run_live, inputs=command, outputs=out_text, show_progress=False)
        with gr.Tab("Training"):
            with gr.Tab("Train Dreambooth"):
                with gr.Box():
                    with gr.Accordion("Train Dreambooth Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
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
                        ```
                        """)
                    train_dreambooth_command = """python /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/dreambooth/train_dreambooth.py \\
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
        --use_8bit_adam"""
                    command = gr.Textbox(show_label=False, lines=16, value=train_dreambooth_command)
                    out_text = gr.Textbox(show_label=False, lines=5)
                    btn_run_live = gr.Button("Train Dreambooth")
                    btn_run_live.click(run_live, inputs=command, outputs=out_text, show_progress=False)
            with gr.Tab("Train LoRA"):
                with gr.Box():
                    with gr.Accordion("Train Lora Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
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
                        ```
                        """)
                    train_lora_command = """python /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/lora/train_dreambooth_lora.py \\
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
        --use_8bit_adam"""
                    command = gr.Textbox(show_label=False, lines=16, value=train_lora_command)
                    out_text = gr.Textbox(show_label=False, lines=5)
                    btn_run_live = gr.Button("Train Lora")
                    btn_run_live.click(run_live, inputs=command, outputs=out_text, show_progress=False)
        with gr.Tab("Convert"):
            with gr.Group():
                with gr.Box():
                    with gr.Accordion("Convert Diffusers to Original Stable Diffusion Arguments", open=False):
                        gr.Markdown(
                        """
                        ```py
                        --model_path /content/stable-diffusion-webui/extensions/stable-diffusion-webui-diffusers/output_dir \\
                        --checkpoint_path /content/stable-diffusion-webui/models/Stable-diffusion/parkminyoung.ckpt
                        ```
                        """)
                    convert_command = """python /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/dreambooth/convert_diffusers_to_original_stable_diffusion.py \\
            --model_path /content/stable-diffusion-webui/extensions/stable-diffusion-webui-diffusers/output_dir \\
            --checkpoint_path /content/stable-diffusion-webui/models/Stable-diffusion/parkminyoung.ckpt"""
                    convert = gr.Textbox(show_label=False, lines=3, value=convert_command)
                    btn_static = gr.Button("Convert Diffusers to Original Stable Diffusion")
                    btn_static.click(run_static, inputs=convert, outputs=out_text, show_progress=False)
                with gr.Box():
                    with gr.Accordion("Remove Dreambooth Output Directory", open=False):
                        gr.Markdown(
                        """
                        ```py
                        rm -rf /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/dreambooth/output_dir/*
                        ```
                        """)
                        rm_command = """rm -rf /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/dreambooth/output_dir/*"""
                        rm = gr.Textbox(show_label=False, lines=1, value=rm_command)
                        btn_static = gr.Button("Remove Dreambooth Output Directory")
                        btn_static.click(run_static, inputs=rm, outputs=out_text, show_progress=False)
                out_text = gr.Textbox(show_label=False)
            with gr.Group():
                with gr.Box():
                    with gr.Accordion("Copy Lora to Additional Network", open=False):
                        gr.Markdown(
                        """
                        ```py
                        cp /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/lora/output_dir/pytorch_lora_weights.safetensors \\
                        /content/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/parkminyoung.safetensors
                        ```
                        """)
                    cp_command = """cp /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/lora/output_dir/pytorch_lora_weights.safetensors \\
        /content/stable-diffusion-webui/extensions/sd-webui-additional-networks/models/lora/parkminyoung.safetensors"""
                    cp = gr.Textbox(show_label=False, lines=2, value=cp_command)
                    btn_static = gr.Button("Copy Lora to Additional Network")
                    btn_static.click(run_static, inputs=cp, outputs=out_text, show_progress=False)
                with gr.Box():
                    with gr.Accordion("Remove Lora Output Directory", open=False):
                        gr.Markdown(
                        """
                        ```py
                        rm -rf /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/lora/output_dir/*
                        ```
                        """)
                    rm_command = """rm -rf /content/stable-diffusion-webui/extensions/stable-diffusion-webui-terminal/training/lora/output_dir/*"""
                    rm = gr.Textbox(show_label=False, lines=1, value=rm_command)
                    btn_static = gr.Button("Remove Lora Output Directory")
                    btn_static.click(run_static, inputs=rm, outputs=out_text, show_progress=False)
                out_text = gr.Textbox(show_label=False)
        with gr.Tab("Tests"):
            with gr.Box():
                command = gr.Textbox(show_label=False, lines=1, placeholder="command")
                out_text = gr.Textbox(show_label=False)
                btn_timeout_test = gr.Button("timeout test")
                btn_timeout_test.click(timeout_test, inputs=command, outputs=out_text, show_progress=False)
                btn_clear = gr.Button("clear")
                btn_clear.click(clear_out_text, inputs=[], outputs=out_text, show_progress=False)
    return (terminal, "Terminal", "terminal"),
script_callbacks.on_ui_tabs(on_ui_tabs)
