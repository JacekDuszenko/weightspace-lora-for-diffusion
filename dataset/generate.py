from diffusers import DiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(model_id,).to("cuda")
pipe.load_lora_weights("out")

pipe.safety_checker = None

prompt = "a photo of sks dog"
images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=5).images
for i, image in enumerate(images):
    image.save(f"dog-bucket_{i}.png")
