from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import gc
import gradio as gr
import random
import os
from dotenv import load_dotenv
from openai import OpenAI
import re

# 自動選擇裝置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 讀取 API Key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# 模型與風格定義
model_name = "digiplay/Sweet-mix_v2.2_flat"
style_prompts = {
    "浴衣": "Yukata",
    "賽博朋克": "cyberpunk",
    "街頭風": "street fashion",
    "學生制服": "school uniform",
}
style_choices = list(style_prompts.keys())

# 載入模型
pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    use_safetensors=True
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# 翻譯中文 Prompt 成英文
def translate_prompt_to_english(chinese_prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "你是一個幫助圖像生成模型翻譯提示詞的助手，請將中文 prompt 轉換英文"},
            {"role": "user", "content": chinese_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def is_english(text):
    return re.match(r'^[a-zA-Z0-9\s.,\-_:;!?()\'"]+$', text.strip()) is not None

# 生成圖片主函數
def generate_images(prompt, use_enhance, enhance_text, use_negative, negative_text,
                    use_custom_seed, custom_seed, height, width, steps, num_images, selected_styles):

    height = int(height)
    width = int(width)

    if not is_english(prompt):
        prompt = translate_prompt_to_english(prompt)

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("高度和寬度必須是8的倍數！")

    base_seed = int(custom_seed) if use_custom_seed else random.randint(0, 2**32 - 1)
    seeds = [base_seed + i for i in range(num_images)]

    final_prompt = prompt
    if use_enhance and enhance_text:
        final_prompt += ", " + enhance_text
    if selected_styles:
        style_texts = [style_prompts[s] for s in selected_styles if s in style_prompts]
        final_prompt = ", ".join(style_texts) + ", " + final_prompt

    final_negative = negative_text if use_negative else None

    gc.collect()
    torch.cuda.empty_cache()

    images = []
    for i in range(num_images):
        generator = torch.Generator(device=device).manual_seed(seeds[i])
        with torch.no_grad():
            result = pipe(
                prompt=final_prompt,
                negative_prompt=final_negative,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=7.5,
                generator=generator
            )
        images.append(result.images[0])

    return images, f"使用的 random seeds: {seeds}"

# 預設值
default_enhance = "masterpiece, ultra high quality, intricate skin details, cinematic lighting"
default_negative = "bad anatomy, blurry, disfigured, poorly drawn hands, extra fingers, mutated hands, low quality, worst quality"

# Gradio UI
with gr.Blocks(css=".gradio-container {background-color: #FAFAFA; padding: 20px;}") as demo:
    gr.Markdown("# 🎨 MajicMIX v6 互動圖像生成器")

    with gr.Row():
        with gr.Column(scale=6):
            selected_styles = gr.CheckboxGroup(choices=style_choices, label="選擇風格", type="value")
            prompt = gr.Textbox(label="Prompt", placeholder="輸入提示詞", lines=3)

            use_enhance = gr.Checkbox(label="加強 Prompt", value=True)
            enhance_text = gr.Textbox(label="加強內容", value=default_enhance)

            use_negative = gr.Checkbox(label="使用 Negative Prompt", value=True)
            negative_text = gr.Textbox(label="Negative Prompt 內容", value=default_negative)

            use_custom_seed = gr.Checkbox(label="自訂 Random Seed", value=False)
            custom_seed = gr.Number(label="指定 seed", value=42)

            height = gr.Dropdown(["512", "768", "1024"], label="高度 Height", value="512")
            width = gr.Dropdown(["512", "768", "1024"], label="寬度 Width", value="512")

            steps = gr.Slider(10, 50, value=20, step=5, label="生成步數 (Steps)")
            num_images = gr.Slider(1, 4, step=1, value=1, label="生成張數")

            generate_btn = gr.Button("🚀 開始生成！")

        with gr.Column(scale=6):
            gallery = gr.Gallery(label="生成結果", columns=2, object_fit="contain", height="auto")
            seed_info = gr.Label(label="使用的 Random Seeds")

    generate_btn.click(
        fn=generate_images,
        inputs=[prompt, use_enhance, enhance_text, use_negative, negative_text,
                use_custom_seed, custom_seed, height, width, steps, num_images, selected_styles],
        outputs=[gallery, seed_info]
    )

    demo.launch(server_port=int(os.getenv("PORT", 7860)), server_name="0.0.0.0", debug=True)
    print("Service started and listening on port", os.getenv("PORT", 7860))
