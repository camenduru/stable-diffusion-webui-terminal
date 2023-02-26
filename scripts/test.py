import os, time
import gradio as gr

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
    with gr.Blocks() as test:
      b1=gr.Button()
      command = gr.Textbox(max_lines=1, placeholder="command")
      text_out1 = gr.Textbox()
      b1.click(run, [command], [text_out1])

      b2=gr.Button()
      command = gr.Textbox(max_lines=1, placeholder="command")
      text_out2 = gr.Textbox()
      b2.click(counter, [], [text_out2])
    return (test, "test", "test")
 
# test.queue()
# test.dependencies[0]["show_progress"] = False
# test.launch(debug=True, share=True, inline=False)

script_callbacks.on_ui_tabs(on_ui_tabs)
