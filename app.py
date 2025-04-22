import os
import shutil
import gradio as gr
from transformers import ReactCodeAgent, HfEngine, Tool
import pandas as pd

from gradio import Chatbot
from streaming import stream_to_gradio
from huggingface_hub import login
from gradio.data_classes import FileData

login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

llm_engine = HfEngine("meta-llama/Llama-3.3-70B-Instruct")

agent = ReactCodeAgent(
    tools=[],
    llm_engine=llm_engine,
    additional_authorized_imports=["numpy", "pandas", "matplotlib", "seaborn","scipy","sklearn"],
    max_iterations=10,
)

base_prompt = """You are an expert full stack data analyst.
You are given a data file and the data structure below.
The data file is passed to you as the variable data_file, it is a pandas dataframe, you can use it directly.
DO NOT try to load data_file, it is already a dataframe pre-loaded in your python interpreter!
When plotting using matplotlib/seaborn save the figures to the (already existing) folder'./figures/': take care to clear 
each figure with plt.clf() before doing another plot.
When plotting make the plots as visually appealing as possible. Same with tables, charts, or anything else.

Use the data file to answer the question or perform a task below.

Structure of the data:
{structure_notes}

Question/Problem:
"""

example_notes="""What is the survival rate by class?"""

def get_images_in_directory(directory):
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}

    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    return image_files

def interact_with_agent(file_input, additional_notes):
    shutil.rmtree("./figures")
    os.makedirs("./figures")

    data_file = pd.read_csv(file_input)
    data_structure_notes = f"""- Description (output of .describe()):
    {data_file.describe()}
    - Columns with dtypes:
    {data_file.dtypes}"""

    prompt = base_prompt.format(structure_notes=data_structure_notes)

    if additional_notes and len(additional_notes) > 0:
        prompt += additional_notes

    messages = [gr.ChatMessage(role="user", content=additional_notes)]
    yield messages + [
        gr.ChatMessage(role="assistant", content="⏳ _Starting task..._")
    ]

    plot_image_paths = {}
    for msg in stream_to_gradio(agent, prompt, data_file=data_file):
        messages.append(msg)
        for image_path in get_images_in_directory("./figures"):
            if image_path not in plot_image_paths:
                image_message = gr.ChatMessage(
                    role="assistant",
                    content=FileData(path=image_path, mime_type="image/png"),
                )
                plot_image_paths[image_path] = True
                messages.append(image_message)
        yield messages + [
            gr.ChatMessage(role="assistant", content="⏳ _Still processing..._")
        ]
    yield messages


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.yellow,
    )
) as demo:
    gr.Markdown("""# Data Analyst (ReAct Code Agent) 📊🤔 
                
**Who am I?** 
I'm your personal Data Analyst built on top of Llama-3.3-70B-Instruct model and the ReAct (Reasoning and Acting) framework.
I break down the task step-by-step until I reach an answer/solution.
Along the way I share my thoughts, actions (Python code blobs), and observations.
I come packed with pandas, numpy, sklearn, matplotlib, seaborn, and more!
                
**Instructions**
1. Drop or upload a `.csv` file below.
2. Ask a question or give it a task.
3. **Watch the AI Agent think, act, and observe until final answer.
\n**For an example, click on the example at the bottom of page to auto populate.**""")

    file_input = gr.File(label="Drop/upload a .csv file to analyze")
    text_input = gr.Textbox(
        label="Ask a question or give it a task."
    )
    submit = gr.Button("Run", variant="primary")
    gr.Examples(
        examples=[["./example/titanic.csv", example_notes]],
        inputs=[file_input, text_input],
        cache_examples=False,
        label='Click on an example below.'
    )
    chatbot = gr.Chatbot(
        label="Data Analyst Agent",
        type="messages",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
        ),
        height = 1000
    )

    submit.click(interact_with_agent, [file_input, text_input], [chatbot])

if __name__ == "__main__":
    demo.launch(share=True)
