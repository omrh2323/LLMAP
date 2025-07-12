from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import os
import uuid
import datetime
import time
import logging
from redis import asyncio as aioredis
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
from typing import List, Generator, Dict, Any, Optional
import asyncio
import json
import threading
import contextvars
import re
import backoff
import gc
import hashlib
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_url, get_hf_file_metadata
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError, HfHubHTTPError
import psutil

# === Configuration ===
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-alpha")
API_KEY = os.getenv("API_KEY", "default_secret_key")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "120"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MAX_CONCURRENT_STREAMS = int(os.getenv("MAX_CONCURRENT_STREAMS", "10"))
GPU_ENABLED = torch.cuda.is_available() and os.getenv("USE_GPU", "true").lower() == "true"
DEVICE = "cuda" if GPU_ENABLED else "cpu"
DTYPE = torch.float16 if GPU_ENABLED else torch.float32
HF_TOKEN = os.getenv("HF_TOKEN", None)
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

# === System Prompt ===
SYSTEM_PROMPT = (
    "You are AyAI, a specialized AI assistant developed for the AySearch project. "
    "Your mission is to evaluate web content in terms of safety, quality, and meaning. "
    "When necessary, provide short summaries and deliver objective, accurate feedback to the user."
)

# === Analysis System Prompt (English) ===
ANALYSIS_PROMPT = (
    "Perform a comprehensive analysis of the following text and provide ONLY JSON output with these keys:\n"
    "- \"summary\": concise summary (max 30 words)\n"
    "- \"is_safe\": boolean indicating safety (check for violence, hate, explicit content)\n"
    "- \"language\": ISO 639-1 language code\n"
    "- \"keywords\": top 5 keywords (array)\n"
    "- \"quality_score\": content quality rating 1-10\n"
    "- \"sentiment\": sentiment analysis (positive, negative, neutral)\n"
    "- \"readability\": readability score 1-10\n"
    "Text: {text}"
)

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(request_id)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_service.log")
    ]
)
logger = logging.getLogger("AyAI-Service")

# Context variable for request tracking
request_id_ctx = contextvars.ContextVar("request_id", default="system")

class RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_ctx.get()
        return True
logger.addFilter(RequestIDFilter())

# === Global State ===
model = None
tokenizer = None
model_loading_lock = asyncio.Lock()
model_loaded = asyncio.Event()
model_loading_failed = asyncio.Event()
redis_client = None

# === Instrumentation Setup ===
instrumentator = Instrumentator()

# === Enhanced Download Functions ===
def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()

def download_with_resume(url: str, local_path: str, expected_hash: str = None, max_retries: int = MAX_RETRIES) -> bool:
    """Robust file download with resume, retry, and hash verification"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            # Check existing file
            file_exists = False
            if os.path.exists(local_path):
                file_exists = True
                if expected_hash:
                    file_hash = calculate_file_hash(local_path)
                    if file_hash == expected_hash:
                        logger.info(f"File already exists and hash matches: {local_path}")
                        return True
                    logger.warning(f"File exists but hash mismatch: {file_hash} vs {expected_hash}, redownloading")
                
                # Resume download
                headers = {}
                file_size = os.path.getsize(local_path)
                headers['Range'] = f'bytes={file_size}-'
            else:
                file_size = 0
            
            with requests.get(url, stream=True, headers=headers, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0)) + file_size
                
                # Append mode if resuming, otherwise write new
                mode = 'ab' if file_size > 0 else 'wb'
                with open(local_path, mode) as f, tqdm(
                    total=total_size, 
                    unit='B', 
                    unit_scale=True, 
                    desc=os.path.basename(local_path),
                    initial=file_size
                ) as progress_bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))
            
            # Verify hash if provided
            if expected_hash:
                file_hash = calculate_file_hash(local_path)
                if file_hash != expected_hash:
                    raise ValueError(f"Hash mismatch after download: {file_hash} vs {expected_hash}")
            
            return True
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.error(f"Download attempt {attempt+1} failed: {str(e)}")
            if file_exists and os.path.exists(local_path):
                os.remove(local_path)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All download attempts failed for {url}")
                return False

def download_model_file(repo_id: str, filename: str) -> str:
    """Download a single model file with robust error handling"""
    # Get file metadata to get the expected hash
    try:
        url = hf_hub_url(repo_id, filename, token=HF_TOKEN)
        file_metadata = get_hf_file_metadata(url, token=HF_TOKEN)
        expected_hash = file_metadata.commit_hash
    except Exception as e:
        logger.warning(f"Could not get file metadata for {filename}: {str(e)}. Skipping hash verification.")
        expected_hash = None

    # Create local path
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_CACHE_DIR, filename)
    
    # Download with resume and hash verification
    success = download_with_resume(url, local_path, expected_hash)
    if not success:
        raise RuntimeError(f"Failed to download {filename} after multiple attempts")
    
    return local_path

# === Robust Model Loading ===
@backoff.on_exception(backoff.expo, 
                      (OSError, RuntimeError, ConnectionError,
                       RepositoryNotFoundError, RevisionNotFoundError, HfHubHTTPError,
                       torch.cuda.OutOfMemoryError, ValueError), 
                      max_tries=MAX_RETRIES, 
                      jitter=backoff.full_jitter,
                      logger=logger,
                      giveup=lambda e: isinstance(e, RepositoryNotFoundError))
def safe_model_load(model_name: str, **kwargs) -> torch.nn.Module:
    """Safely load model with retry logic and fallback"""
    try:
        # First try with GPU if enabled
        if GPU_ENABLED:
            logger.info("Attempting GPU model load")
            return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        
        # Fallback to CPU
        logger.warning("Falling back to CPU model load")
        kwargs['device_map'] = 'cpu'
        kwargs['torch_dtype'] = torch.float32
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory, trying CPU fallback")
        kwargs['device_map'] = 'cpu'
        kwargs['torch_dtype'] = torch.float32
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

# === Memory Management ===
def log_memory_usage():
    """Log detailed memory usage information"""
    if GPU_ENABLED and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i} Memory: Allocated={mem_alloc:.2f}GB, Reserved={mem_reserved:.2f}GB")
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"RAM Usage: RSS={mem_info.rss/1024**2:.2f}MB, VMS={mem_info.vms/1024**2:.2f}MB")

# === App Lifecycle ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, model, tokenizer
    
    # Initialize Redis
    try:
        redis_client = aioredis.from_url(REDIS_URL, encoding="utf8", decode_responses=True)
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Redis connection failed: {str(e)}")
        redis_client = None
    
    # Start model loading
    asyncio.create_task(load_model())
    
    # Expose metrics endpoint
    instrumentator.expose(app)
    
    yield
    
    # Cleanup resources
    cleanup_success = True
    if model is not None:
        try:
            del model
            model = None
            logger.info("Model resources released")
        except Exception as e:
            logger.error(f"Model cleanup failed: {str(e)}")
            cleanup_success = False
    
    if tokenizer is not None:
        try:
            del tokenizer
            tokenizer = None
            logger.info("Tokenizer released")
        except Exception as e:
            logger.error(f"Tokenizer cleanup failed: {str(e)}")
            cleanup_success = False
    
    if redis_client is not None:
        try:
            await redis_client.close()
            redis_client = None
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Redis close failed: {str(e)}")
            cleanup_success = False
    
    # Force cleanup
    gc.collect()
    if GPU_ENABLED:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.ipc_collect()
            logger.info("CUDA cache cleared")
        except Exception as e:
            logger.error(f"CUDA cleanup failed: {str(e)}")
            cleanup_success = False
    
    log_memory_usage()
    logger.info(f"Cleanup completed {'successfully' if cleanup_success else 'with errors'}")

# Create app with lifespan
app = FastAPI(lifespan=lifespan, title="AyAI Service", version="2.4")

# Instrument the app
instrumentator.instrument(app)

# === Model Loading with Integrity Check ===
async def load_model():
    global model, tokenizer
    async with model_loading_lock:
        if model_loaded.is_set() or model_loading_failed.is_set():
            return
            
        logger.info("Starting model loading...")
        start_time = time.time()
        
        try:
            log_memory_usage()
            
            # Attempt to load tokenizer first
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAME, 
                    token=HF_TOKEN,
                    cache_dir=MODEL_CACHE_DIR
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Tokenizer loading failed: {str(e)}")
                raise
            
            # Configuration for model loading
            load_kwargs = {
                "torch_dtype": DTYPE,
                "low_cpu_mem_usage": True,
                "token": HF_TOKEN,
                "cache_dir": MODEL_CACHE_DIR
            }
            
            if GPU_ENABLED:
                # Use smarter device mapping for multi-GPU systems
                if torch.cuda.device_count() > 1:
                    load_kwargs["device_map"] = "balanced_low_0"
                    logger.info("Using multi-GPU balanced loading")
                else:
                    load_kwargs["device_map"] = "auto"
            
            # Load model with retry logic
            model = safe_model_load(MODEL_NAME, **load_kwargs)
            
            # Run a test inference to validate model
            try:
                test_input = tokenizer("Test input for model validation", return_tensors="pt").to(model.device)
                with torch.no_grad():
                    model(**test_input)
                logger.info("Model validation successful")
            except Exception as e:
                logger.error(f"Model validation failed: {str(e)}")
                raise RuntimeError("Model validation failed")
            
            # Use BetterTransformer if available
            if GPU_ENABLED and hasattr(model, "to_bettertransformer"):
                try:
                    model = model.to_bettertransformer()
                    logger.info("Using BetterTransformer optimization")
                except Exception as e:
                    logger.warning(f"BetterTransformer failed: {str(e)}")
            
            model_loaded.set()
            logger.info(f"Model loaded in {time.time() - start_time:.2f}s | Device: {model.device}")
            log_memory_usage()
            
        except Exception as e:
            logger.critical(f"Model loading failed: {str(e)}")
            # Attempt to free resources
            if 'model' in locals():
                try:
                    del model
                except:
                    pass
            model = None
            model_loading_failed.set()
            model_loaded.set()

# === Request Models ===
class PromptInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2048)
    max_tokens: int = Field(default=500, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    stream: bool = Field(default=False)

class BatchPromptInput(BaseModel):
    prompts: List[str] = Field(..., min_items=1, max_items=32)
    max_tokens: int = Field(default=500, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)

class AnalyzeInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    detailed: bool = Field(default=False)

# === API Key Validation ===
async def validate_api_key(request: Request):
    if request.headers.get("x-api-key") != API_KEY:
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(f"Invalid API key from {client_ip}")
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# === Generation Utilities ===
def format_prompt(prompt: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAyAI:"

def generate_response(prompt: str, max_tokens: int, temperature: float) -> str:
    if model_loading_failed.is_set():
        return "Error: Model failed to load"

    formatted_prompt = format_prompt(prompt)
    
    try:
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True, 
            max_length=4096
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        # Extract only the generated text
        response_start = inputs.input_ids.shape[-1]
        response = outputs[0][response_start:]
        return tokenizer.decode(response, skip_special_tokens=True).strip()
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory during generation")
        return "Error: Model overloaded, please try again"
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return "Error: Generation failed"

async def generate_stream(prompt: str, max_tokens: int, temperature: float) -> Generator[str, None, None]:
    if model_loading_failed.is_set():
        yield json.dumps({"error": "Model failed to load"}) + "\n"
        return

    formatted_prompt = format_prompt(prompt)
    
    inputs = tokenizer(
        [formatted_prompt], 
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True,
        timeout=300
    )
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Run generation in separate thread
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream tokens as they're generated
    try:
        async for token in streamer:
            yield json.dumps({"token": token}) + "\n"
            await asyncio.sleep(0.001)
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield json.dumps({"error": "Stream interrupted"}) + "\n"

# === Batch Processing ===
def process_batch(prompts: List[str], max_tokens: int, temperature: float) -> List[str]:
    if model_loading_failed.is_set():
        return ["Error: Model failed to load"] * len(prompts)

    formatted_prompts = [format_prompt(p) for p in prompts]
    
    try:
        inputs = tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        # Extract only the generated text for each prompt
        responses = []
        for i in range(len(prompts)):
            response_start = inputs.input_ids[i].shape[0]
            response = outputs[i][response_start:]
            responses.append(tokenizer.decode(response, skip_special_tokens=True).strip())
        
        return responses
    
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory during batch processing")
        return ["Error: System overloaded, please try again"] * len(prompts)
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        return ["Error: Processing failed"] * len(prompts)

# === Enhanced Analysis Functions ===
def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extraction from model response"""
    # Try to find complete JSON structure
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            # Attempt to fix common formatting issues
            try:
                # Fix unescaped quotes
                fixed_json = re.sub(r'([^\\])\\"', r'\1\\\\"', json_match.group())
                # Fix trailing commas
                fixed_json = re.sub(r',\s*([}\]])', r'\1', fixed_json)
                return json.loads(fixed_json)
            except:
                pass
    
    # Fallback: Find JSON-like structures
    try:
        # Attempt to parse the entire response as JSON
        return json.loads(response)
    except:
        # Try to find the first valid JSON object
        start_idx = response.find('{')
        while start_idx != -1:
            end_idx = start_idx + 1
            brace_count = 1
            while end_idx < len(response) and brace_count > 0:
                if response[end_idx] == '{':
                    brace_count += 1
                elif response[end_idx] == '}':
                    brace_count -= 1
                end_idx += 1
            
            if brace_count == 0:
                try:
                    return json.loads(response[start_idx:end_idx])
                except:
                    pass
            
            start_idx = response.find('{', start_idx + 1)
    
    return None

def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text and return structured JSON"""
    if model_loading_failed.is_set():
        return {"error": "Model failed to load", "status": "unavailable"}

    formatted_prompt = ANALYSIS_PROMPT.format(text=text)
    
    try:
        # Get model response
        response = generate_response(
            prompt=formatted_prompt,
            max_tokens=600,
            temperature=0.3
        )
        
        # Extract JSON part
        result = extract_json_from_response(response)
        
        if not result:
            logger.error(f"JSON output not found in: {response[:200]}...")
            return {"error": "Analysis output could not be processed"}
        
        # Validate required keys
        required_keys = ["summary", "is_safe", "language", "keywords", "quality_score"]
        for key in required_keys:
            if key not in result:
                logger.warning(f"Missing key in analysis: {key}")
                result[key] = None
        
        return result
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"error": "Error during analysis"}

def enhanced_analysis(base_result: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Enhance the base analysis with additional metrics"""
    if "error" in base_result:
        return base_result
    
    # Add processing metadata
    base_result["analysis_timestamp"] = datetime.datetime.utcnow().isoformat()
    base_result["word_count"] = len(text.split())
    base_result["char_count"] = len(text)
    
    # Calculate keyword metrics if possible
    if "keywords" in base_result and isinstance(base_result["keywords"], list):
        text_lower = text.lower()
        keyword_metrics = []
        total_words = len(text.split())
        
        for keyword in base_result["keywords"]:
            if isinstance(keyword, str):
                count = text_lower.count(keyword.lower())
                density = (count / total_words * 100) if total_words > 0 else 0
                keyword_metrics.append({
                    "keyword": keyword,
                    "count": count,
                    "density": f"{density:.2f}%"
                })
        
        base_result["keyword_metrics"] = keyword_metrics
    
    return base_result

# === Rate Limiting Middleware ===
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Generate request ID and set in context
    req_id = str(uuid.uuid4())
    request_id_ctx.set(req_id)
    request.state.request_id = req_id
    
    # Skip rate limiting for health and info endpoints
    if request.url.path in ["/health", "/info"]:
        return await call_next(request)
    
    # Apply rate limiting
    endpoint = request.url.path
    client_ip = request.client.host if request.client else "unknown"
    
    # Default cost
    cost = 1
    
    try:
        # Adjust cost based on endpoint
        if endpoint == "/batch" and request.method == "POST":
            # Read body for batch requests
            body = await request.body()
            try:
                data = json.loads(body)
                cost = min(len(data.get("prompts", [])), 10)
            except:
                cost = 10
            # Restore body for downstream processing
            request._body = body
        elif endpoint == "/analyze" and request.method == "POST":
            cost = 3  # Higher cost for analysis
        
        # Apply rate limit if Redis is available
        if redis_client:
            key = f"rate_limit:{client_ip}:{endpoint}"
            current = await redis_client.get(key)
            current = int(current) if current else 0
            
            if current + cost > MAX_REQUESTS_PER_MINUTE:
                logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}")
                raise HTTPException(status_code=429, detail="Too many requests")
            
            # Update rate limit
            await redis_client.incrby(key, cost)
            if current == 0:
                await redis_client.expire(key, 60)
        
        # Process request
        response = await call_next(request)
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Middleware error: {str(e)}")
        return await call_next(request)

# === Endpoints ===
@app.post("/ask", dependencies=[Depends(validate_api_key)])
async def ask_llm(
    request: Request, 
    prompt_input: PromptInput, 
    background_tasks: BackgroundTasks
):
    # Wait for model to load
    if not model_loaded.is_set():
        await model_loaded.wait()
    
    req_id = request.state.request_id
    
    if model_loading_failed.is_set():
        raise HTTPException(
            status_code=500,
            detail="Model failed to load, service unavailable"
        )
    
    if prompt_input.stream:
        if len(background_tasks.tasks) >= MAX_CONCURRENT_STREAMS:
            raise HTTPException(
                status_code=429, 
                detail="Too many concurrent streams"
            )
            
        return StreamingResponse(
            generate_stream(
                prompt_input.prompt, 
                prompt_input.max_tokens, 
                prompt_input.temperature
            ),
            media_type="application/x-ndjson"
        )
    
    # Non-streaming response
    response = generate_response(
        prompt_input.prompt, 
        prompt_input.max_tokens, 
        prompt_input.temperature
    )
    
    return {
        "response": response,
        "id": req_id,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.post("/batch", dependencies=[Depends(validate_api_key)])
async def batch_llm(request: Request, batch_input: BatchPromptInput):
    if not model_loaded.is_set():
        await model_loaded.wait()
    
    if model_loading_failed.is_set():
        raise HTTPException(
            status_code=500,
            detail="Model failed to load, service unavailable"
        )
    
    req_id = request.state.request_id
    
    # Process in batches
    all_results = []
    for i in range(0, len(batch_input.prompts), BATCH_SIZE):
        batch = batch_input.prompts[i:i+BATCH_SIZE]
        results = process_batch(
            batch,
            batch_input.max_tokens,
            batch_input.temperature
        )
        all_results.extend(results)
    
    return {
        "responses": all_results,
        "count": len(all_results),
        "batch_id": req_id,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.post("/analyze", dependencies=[Depends(validate_api_key)])
async def analyze_content(
    request: Request, 
    analyze_input: AnalyzeInput,
    background_tasks: BackgroundTasks
):
    if not model_loaded.is_set():
        await model_loaded.wait()
    
    if model_loading_failed.is_set():
        raise HTTPException(
            status_code=500,
            detail="Model failed to load, service unavailable"
        )
    
    req_id = request.state.request_id
    logger.info(f"Analysis request started: {req_id}")
    
    # Perform base analysis
    start_time = time.time()
    analysis_result = analyze_text(analyze_input.text)
    
    # If detailed analysis is requested
    if analyze_input.detailed:
        analysis_result = enhanced_analysis(analysis_result, analyze_input.text)
    
    # Add processing metrics
    processing_time = time.time() - start_time
    analysis_result["processing_time_sec"] = f"{processing_time:.2f}"
    analysis_result["request_id"] = req_id
    
    return {
        "analysis": analysis_result,
        "id": req_id,
        "text_length": len(analyze_input.text),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.get("/info")
async def get_info(request: Request):
    status = "ready" if model_loaded.is_set() else "initializing"
    if model_loading_failed.is_set():
        status = "failed"
    device_info = str(model.device) if model else "N/A"
    
    return {
        "model": MODEL_NAME,
        "status": status,
        "device": device_info,
        "gpu_enabled": GPU_ENABLED,
        "rate_limit": MAX_REQUESTS_PER_MINUTE,
        "batch_size": BATCH_SIZE,
        "max_streams": MAX_CONCURRENT_STREAMS,
        "version": app.version
    }

@app.get("/health")
async def health_check():
    status = "healthy"
    if model_loading_failed.is_set():
        status = "unhealthy"
    elif not model_loaded.is_set():
        status = "initializing"
    
    return {
        "status": status,
        "model_loaded": model_loaded.is_set(),
        "model_loading_failed": model_loading_failed.is_set(),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

# === Launch ===
if __name__ == "__main__":
    # Set Hugging Face environment variables for better stability
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080, 
        workers=1,
        timeout_keep_alive=60
    )