import os
import torch
import streamlit as st
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
import logging
import io
import gc
import pkg_resources

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    logger.info("Checking dependencies...")
    required_packages = {
        'protobuf': '3.20.0',
        'transformers': '4.30.0',
        'diffusers': '0.21.0',
        'accelerate': '0.20.0',
        'safetensors': '0.3.1'
    }
    
    for package, min_version in required_packages.items():
        try:
            version = pkg_resources.get_distribution(package).version
            logger.info(f"{package} version: {version}")
            if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                st.warning(f"{package} version {version} is below recommended version {min_version}")
        except pkg_resources.DistributionNotFound:
            logger.error(f"{package} is not installed")
            st.error(f"Required package {package} is not installed")
            raise RuntimeError(f"Required package {package} is not installed")

def initialize_cuda():
    logger.info("Initializing CUDA...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        logger.info(f"CUDA initialized successfully on device: {torch.cuda.get_device_name()}")
        st.success(f"CUDA initialized on: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA is not available")
        st.warning("CUDA is not available. Running on CPU may be slower.")

@st.cache_resource
def load_pipeline():
    try:
        model_id = "stabilityai/stable-diffusion-3.5-medium"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )
        
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=model_nf4,
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        
        logger.info("Pipeline loaded successfully with quantization.")
        st.success("Model loaded successfully!")
        return pipe
    except Exception as e:
        error_msg = f"Error loading pipeline: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        raise

def generate_images(pipe, prompt, num_inference_steps, guidance_scale):
    try:
        logger.info(f"Generating image with prompt: {prompt}")
        with st.spinner("Generating your images... Please wait."):
            image = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=512
            ).images[0]
        logger.info("Image generation completed successfully")
        st.success("Image generated successfully!")
        return image
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        raise

def main():
    st.set_page_config(page_title="SD 3.5 Medium Image Generator", page_icon="🎨", layout="wide")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
            .main-title {
                text-align: center;
                font-size: 36px;
                color: #ffffff;
                background: linear-gradient(135deg, #ff7eb3, #ff758c);
                padding: 20px;
                border-radius: 10px;
            }
            .sidebar .sidebar-content {
                background-color: #222;
                padding: 20px;
                border-radius: 10px;
            }
            .sidebar .sidebar-content h3 {
                color: #ff758c;
            }
            .block-container {
                background-color: #1e1e1e;
                color: white;
                padding: 30px;
                border-radius: 10px;
            }
            .stButton button {
                background-color: #ff758c !important;
                color: white !important;
                border-radius: 5px !important;
                padding: 10px 20px !important;
                font-size: 16px !important;
                border: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<h1 class="main-title">🎨 Image Generator with Stable Diffusion 3.5 Medium</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Image Generation Settings")
        prompt = st.text_area("Enter your prompt:", "A futuristic city at sunset, cyberpunk style")
        num_inference_steps = st.slider("Inference Steps:", min_value=1, max_value=50, value=40)
        guidance_scale = st.slider("Guidance Scale:", min_value=1.0, max_value=10.0, value=4.5)
        generate_button = st.button("🎨 Generate Image", type="primary")
    
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None
    
    if generate_button:
        try:
            check_dependencies()
            initialize_cuda()
            pipe = load_pipeline()
            image = generate_images(pipe, prompt, num_inference_steps, guidance_scale)
            st.session_state.generated_image = image
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    if st.session_state.generated_image is not None:
        st.markdown("### 🖼️ Generated Image")
        st.image(st.session_state.generated_image, caption="Generated Image", use_container_width=True)
        
        buf = io.BytesIO()
        st.session_state.generated_image.save(buf, format='PNG')
        st.download_button("📥 Download Image", data=buf.getvalue(), file_name="generated_image.png", mime="image/png")

if __name__ == "__main__":
    main()
