import os, time
import gradio as gr
from modules import script_callbacks

def counter():
  total = 10
  print("header")
  yield f"\nheader"
  for i in range(total):
    progress = "#" * (i + 1)
    spaces = " " * (total - i - 1)
    percent = (i + 1) * 10
    print(f"\r[{progress}{spaces}] {percent}%", end='')
    yield f"\r[{progress}{spaces}] {percent}%"
    time.sleep(1)
  print(f"\nfooter")
  yield f"\nfooter"
  
def run(command):
  with os.popen(command) as pipe:
    for line in pipe:
      line = line.rstrip()
      print(line)
      yield line

def on_ui_tabs():
    with gr.Blocks() as colabtest:
      b1=gr.Button()
      command = gr.Textbox(max_lines=1, placeholder="command")
      text_out = gr.Textbox()
      b2=gr.Button()
      b1.click(run, [command], [text_out])
      b2.click(counter, [], [text_out])
      return (colabtest, "colabtest", "colabtest")
script_callbacks.on_ui_tabs(on_ui_tabs)
