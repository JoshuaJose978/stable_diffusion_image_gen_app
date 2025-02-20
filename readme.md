To activate a virtual environment in PowerShell, you need to run the activation script with the correct execution policy. Here's how:

1. First, navigate to your project directory where the venv is located:
```powershell
cd your-project-directory
```

2. Then run the activation script:
```powershell
.\virt_env\Scripts\Activate.ps1
```

If you get a security error about running scripts, you'll need to allow script execution by running:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Once activated, you'll see `(venv)` at the beginning of your PowerShell prompt.

To deactivate the virtual environment when you're done, simply run:
```powershell
deactivate
```


---

### 1. **`num_inference_steps` (Inference Steps)**
   - Controls **how many denoising steps** the model takes to generate the final image.
   - More steps generally lead to a higher-quality image with finer details and better coherence.
   - However, beyond a certain point (usually **30–50 steps** for Stable Diffusion), the improvements become marginal while increasing computation time.
   - **In your code, you’ve set `num_inference_steps=40`**, which is a reasonable balance between quality and speed.

### 2. **`guidance_scale` (Classifier-Free Guidance)**
   - Determines how strongly the model **adheres to the text prompt**.
   - Higher values make the image follow the prompt more closely but may over-constrain the model, making the image look less natural.
   - Lower values give the model more creative freedom but may result in a less accurate interpretation of the prompt.
   - **In your code, `guidance_scale=4.5` is a moderate setting**, allowing for both prompt adherence and some creative variation. Typically:
     - **1.0–3.5**: Very loose, allowing creative freedom but might ignore prompt details.
     - **4.0–8.0**: Balanced (your choice falls in this range).
     - **10+**: Very strict adherence, but images may look unnatural or overprocessed.

----

Steps to run the app in your system

Open a New Terminal

Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`

Run `.\virt_env\Scripts\Activate.ps1`

Run `streamlit run .\app.py`


docker build -t image_gen_app .

docker run -d --gpus all -p 8501:8501 image_gen_app

docker run -d --gpus all -v ~/.huggingface:/root/.huggingface -p 8501:8501 image_gen_app

docker run -d --gpus all -v "${env:USERPROFILE}\.cache\huggingface:/app/cache" -p 8501:8501 image_gen_app

