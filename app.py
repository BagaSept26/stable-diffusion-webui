import gradio as gr
import torch
import os
import time #utk nama file unik
from PIL import image
from diffusers import (
  StableDiffusionPipeline,
  StableDiffusionImg2ImgPipeline,
  StableDiffusionInpaintPipeline,
  #Daftar scheduler
  EulerDiscreteScheduler,
  EulerAncestralDiscreteScheduler,
  DPMSolverMultistepScheduler,
  LMSDiscreteScheduler,
  PNDMScheduler,
  UniPCMultistepScheduler
)
from.diffusers.utils import load_image #utk memuat gambar dari url
#--Konfigurasi global & model --
OUTPUT_DIR= "sd_outputs"
os.makedirs=(OUTPUT_DIR, exist_ok=True)

# Pilihan Model
# Format: "Nama Tampilan": "ID Model di Hugging Face Hub"
AVAILABLE_MODELS = {
    "Stable Diffusion 1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion 2.1 (785px)": "stabilityai/stable-diffusion-v2-1",
    "Stable Diffusion 2.1 base (512px)": "stabilityai/stable-diffusion-v2-1-base",
    "OpenJourney (Midjourney Style)": "prompthero/openjourney",
    #Tambahkan model lain disini
}

DEFAULT_MODEL_ID = AVAILABLE_MODELS["Stable Diffusion 1.5"]

# Pilihan Scheduler/Sampler
    "Euler": EulerDiscreteScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    "DPM++ 2M Karras": DPMSolverMultistepScheduler, # Sering jadi default yang baik
    "LMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UniPC": UniPCMultistepScheduler,
    }

DEFAULT_SCHEDULER_NAME = "DPM++ 2M Karras"


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu"
    print("PERINGATAN: Tidak ada GPU yang terdeteksi oleh PyTorch! Aplikasi akan sangat lambat.")
    
current_pipeline = None
current_model_id_loaded = None

#Placeholder untuk fungsi-fungsi utama (diisi di Gitpod)
def load_model_on_demand(model_id_key, scheduler_name_key):
        global current_pipeline, current_model_id_loaded
        selected_model_id = AVAILABLE_MODELS.get(model_id_key,DEFAULT_MODEL_ID)
        
        if current_pipeline is not None and current_model_id_loaded == selected_model_id:
            #jika model sudah dimuat dan scheduler juga sama (atau bisa diubah tanpa reload)
            #untuk kesederhanaan, reload jika model atau scheduler kunci berubah
            #optimasi lebih lanju
            print(f"Model '{selected_model_id}' sudah dimuat.")
            else:
                print(f"Memuat model: {selected_model_id}...")
                #kosongkan vram sebelum memuat model
                if current_pipeline is not None:
                    del current_pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                #tentukan tipe pipeline berdasarkan model atau kebutuhan (txt2img dasar)
                #utk img2img dan inpaint, pipeline akan diganti di fungsi spesifik
                #pipeline"dasar" yang digunakan utk txt2img
                try:
                    dtype = torch.float16 if device == "cuda" else torch.float32
                    current_pipeline = StableDiffusionPipeline.from_pretrained(
                        selected_model_id,
                        torch_dtype=dtype,
                        use_safetensors=True if "safetensors" in 
                        os.listdir(selected_model_id) or ".safetensors" in selected_model_id else False #cek jika safetensors ada
                        )
                        current_model_id_loaded = selected_model_id
                        print(f"Model '{selected_model_id}'berhasil dimuat ke {device}.")
                except Exception as e:
                    print(f"Gagal memuat model '{selected_model_id}': {e}")
                    #fallback ke model default jika gagal
                    if selected_model_id != DEFAULT_MODEL_ID:
                        return load_model_on_demand(list(AVAILABLE_MODELS.keys())
                        [list(AVAILABLE_MODELS.values()).index(DEFAULT_MODEL_ID)], scheduler_name_key)
                    else: #jika model default juga gagal, ada masalah besar
                            raise e #Re-raise exception jika default gagal
                            
                # Set scheduler
                SelectedSchedulerClass = AVAILABLE_SCHEDULERS.get(scheduler_name_key, AVAILABLE_SCHEDULERS[DEFAULT_SCHEDULER_NAME])
                current_pipeline.scheduler = 
                    SelectedSchedulerClass.from_config(current_pipeline.scheduler.config, use_karras_sigmas=True if "Karras" in scheduler_name_key else False)
                    
                current_pipeline = current_pipeline.to(device)
                
                #optimasi (jika diperlukan dan library terinstall
                #if device == "cuda":
                #   try:
                #       current_pipeline.enable_xformers_memory_efficient_attention()
                #       print("xformers memory efficient attention diaktifkan.")
                #   except ImportError:
                #       print("xformers tidak terinstal. pertimbangkan untuk menginstallnya utk performa lebih baik.")
                #   except Exception as e:
                #       print(f"Gagal mengaktifkan xformers: {e}")
                #   # current_pipeline.enable_attention_slicing() #alternatif jika xformers tidak ada
                
                print(f"Scheduler '{scheduler_name_key}'diatur.")
                return f"Model: {model_id_key} & Scheduler: {scheduler_name_key} siap."
   
   def generate_txt2img_main(model_key, scheduler_key, prompt, neg_prompt, width, height, steps, cfg_scale, seed_val):
        """Fungsi inti untuk Text-to-Image."""
        print(f"\n--- Txt2Img Request ---")
        print(f"Model: {model_key}, Scheduler: {scheduler_key}")
        print(f"Prompt: {prompt}")
        print(f"Negative Prompt:{neg_prompt}")
        print(f"Resolusi: {width}x{height}, Steps: {steps}, CFG: {cfg_scale}, Seed: {seed_val}")
        
        status_load = load_model_on_demand(model_key,scheduler_key) #pastikan scheduler model dimuat
        print(status_load)
        
        if current_pipeline is None or not isinstance(current_pipeline, StableDiffusionPipeline):
            #jika pipeline bkn tipe yang benar, muat ulang ( seharusnya ditangati load_model_on_demand)
            #atau, ubah current_pipeline menjadi tipe yang benar
            #utk kesederhanaan, asumsikan load_model_on_demand mentiapkan pipeline txt2img dassar
            print("Pipeline dasar (txt2img) sedang digunakan/dimuat.")
            
        #gunakan seed jika diberikan, jika tidak random
        generator = torch.Generator(device=device)
        if seed_val is not None and seed_val != -1 and str(seed_val).strip()  != "":
            seed = int(seed_val)
            generator = generator.manual_seed(seed)
        else:
            seed = generator.seed() #dapatkan seed random yang digunakan
        
        try:
            start_time = time.time()
            with.torch.inference_mode(): #optimasi tambahan
                image = current_pipeline(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=float(cfg_scale),
                generator=generator
                ).images[0]
            end_time = time.time()
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(OUTPUT_DIR,f"txt2img_{timestamp}_seed{seed}.png")
            image.save(filename)
            
            info_text = f"Seed: {seed}\nModel: {model_key}\nScheduler: {scheduler_key}\nSteps: {steps}, CFG: {cfg_scale}\nTime: {end_time - start_time:.2f}s\nSaved: {filename}"
            print(f"Gambar txt2img berhasill dibuat. Info: {info_text}")
            return image, info_text
            
        except Exception as e:
            print(f"Error saat txt2img: {e}")
            torch.cuda.empty_cache() # bersiihkan cache jika error
            return None, f"Error: {str(e)}"
        #finally:
            #tidak perlu del pipeline disini karna mempertahankan request berikutnya
            #kosongkan cache saja jika VRAM terbatas antar run 
            # if torch.cuda.is_available():
            # torch.cuda.empty_cache():
            
            
# -- UI Gradio ---
#dikembangkan lebih lanjut menggunakan gitpod
def create_gradio_ui():
    with gr.Blocks(css="body {font-family: sans-serif;}") as demo:
        gr.Markdown("# Stable Diffusion Web UI (via hugging face spaces)")
        gr.Markdown(f"Device: `{device}`. pastikan GPU aktif di HF Spaces Settings.")
        with gr.Row():
            model_dropdown = gr.Dropdown(label="Pilih Model Dasar", choices=list(AVAILABLE_MODELS.keys()),value=list(AVAILABLE_MODELS.keys())[0])
            scheduler_dropdown = gr.Dropdown(label="PIlih Sampler(Scheduler)", choices=list(AVAILABLE_SCHEDULERS.keys()),value=DEFAULT_SCHEDULER_NAME)
            
            #tombol utk memat model secara eksplisit jika diperlukan(atau otomatis saat generate)
            #status_load_display = gr.Textbox(label="Status Model", interactive=False)
            #load_model_button = gr.Button("Load Model & Scheduler Pilihan")
            #load_model_button.click(load_model_on_demand, inputs=[model_dropdown, scheduler_dropdown], outputs=[status_load_display])
            
            with gr.Tabs():
                with gr.TabItem("Text-to-Image"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            txt2img_prompt = gr.Textbox(label="Prompt",lines=4, placeholder="A beautiful sunset over a cyberpunk city...")
                            txt2img_neg_prompt = gr.Textbox(label="Negative Prompt", lines=2, placeholder="ugly,blurry,watermark,text,extra fingers....")
                            with gr.Column(scale=1):
                                txt2img_width = gr.Slider(label="Width",minimum=256, maximum=1024, value=512, step=64)
                                txt2img_height = gr.Slider(label="Height",minimum=256, maximum=1024, value=512, step=64)
                                txt2img_steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=150, value=25, step=1)
                                txt2img_cfg_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=30.0, value=7.5, step=0.5)
                                txt2img_seed = gr.Number(label="Seed (-1 untuk random)", value=-1)
                                txt2img_generate_btn = gr.Button("Generate Image", variant="primary")
                            with gr.Row():
                                txt2img_output_image = gr.Image(label="Generated Image", type="pil", height="512") #PIL untk re-use
                                txt2img_output_info = gr.Textbox(label="Generation Info", lines=5, interactive=False)
                            txt2img_generate_btn.click(
                                generate_txt2img_main, 
                                inputs=[model_dropdown, 
                                        scheduler_dropdown, 
                                        txt2img_prompt, 
                                        txt2img_neg_prompt, 
                                        txt2img_width, 
                                        txt2img_height, 
                                        txt2img_steps, 
                                        txt2img_cfg_scale, 
                                        txt2img_seed],
                                outputs=[txt2img_output_image,
                                         txt2img_output_info]
                            )
                            
                            with gr.TabItem("Image-to-Image"):
                                gr.Markdown("Fitur Image-to-Image akan ditambahkan disini.")
                                #placeholder utk input output img2img
                                
                            with gr.TabItem("InPainting"):
                                gr.Markdown("Fitur InPainting akan ditambahkan disini.")
                                #placeholder utk input output inpainting
                                
                            with gr.TabItem("Pengaturan & Info"):
                                gr.Markdown(f"**Model Tersedia:**")
                                for k,v in AVAILABLE_MODELS.items():
                                    gr.Markdown(f"- {k}: `{v}`")
                                gr.Markdown(f"**Sampler Tersedia:**")
                                for k in AVAILABLE_SCHEDULERS.keys():
                                    gr.Markdown(f"- {k}")
                                # gr.Markdown("Token Hugging Face harus di-set sebagai secret `HF_TOKEN` di Settings Space ini jika menggunakan model private atau ingin push ke Hub.")
    return demo

if__name__ == "__main__":
    #muat model default saat aplikasi dimulai (opsional, bisa juga on-demand)
    # print(load_model_on_demand(list(AVAILABLE_MODELS.keys())[0],DEFAULT_SCHEDULER_NAME))
    
    webui = create_gradio_ui()
    webui.launch() #HF Spaces akan menangani eksposure.
#server_name="0.0.0.0" mungkin tidak perlu.
