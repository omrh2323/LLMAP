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

# === Configuration ===
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
API_KEY = os.getenv("API_KEY", "default_secret_key")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "120"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MAX_CONCURRENT_STREAMS = int(os.getenv("MAX_CONCURRENT_STREAMS", "10"))
GPU_ENABLED = torch.cuda.is_available() and os.getenv("USE_GPU", "true").lower() == "true"
DEVICE = "cuda" if GPU_ENABLED else "cpu"
DTYPE = torch.float16 if GPU_ENABLED else torch.float32
PORT = int(os.getenv("PORT", 8080))  # Port configürasyonu eklendi

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
    
    # Cleanup resources with null checks
    if model is not None:
        if GPU_ENABLED:
            with torch.no_grad():
                torch.cuda.empty_cache()
        del model
        model = None
        logger.info("Model resources released")
    
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
        logger.info("Tokenizer released")
    
    if redis_client is not None:
        await redis_client.aclose()  # Düzeltildi: aclose() kullanılıyor
        redis_client = None
        logger.info("Redis connection closed")
    
    # Force garbage collection
    gc.collect()
    if GPU_ENABLED:
        torch.cuda.empty_cache()

# Create app with lifespan
app = FastAPI(lifespan=lifespan, title="AyAI Service", version="2.1")

# Instrument the app
instrumentator.instrument(app)

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

# === Model Loading with Backoff ===
@backoff.on_exception(backoff.expo, 
                      (OSError, RuntimeError, ConnectionError), 
                      max_tries=5, 
                      jitter=backoff.full_jitter,
                      logger=logger)
async def safe_model_load(pretrained_model_name_or_path: str, **kwargs):
    """Safely load model with retry logic"""
    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)

async def load_model():
    global model, tokenizer
    async with model_loading_lock:
        if model_loaded.is_set() or model_loading_failed.is_set():
            return
            
        logger.info("Starting model loading...")
        start_time = time.time()
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Configuration for model loading
            load_kwargs = {
                "torch_dtype": DTYPE,
                "low_cpu_mem_usage": True,
            }
            
            if GPU_ENABLED:
                load_kwargs["device_map"] = "auto"
            
            # Load model with retry logic
            model = await safe_model_load(MODEL_NAME, **load_kwargs)
            
            # Use BetterTransformer if available
            if GPU_ENABLED and hasattr(model, "to_bettertransformer"):
                model = model.to_bettertransformer()
                logger.info("Using BetterTransformer optimization")
            
            model_loaded.set()
            logger.info(f"Model loaded in {time.time() - start_time:.2f}s | Device: {model.device}")
            
        except Exception as e:
            logger.critical(f"Model loading failed: {str(e)}")
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
    detailed: bool = Field(default=False)  # For enhanced analysis

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
        for token in streamer:
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
    # Port çakışmasını önlemek için yeniden kullanım izni
    config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=PORT,
        workers=1,
        timeout_keep_alive=60,
        reuse_port=True  # Port yeniden kullanımı için
    )
    server = uvicorn.Server(config)
    server.run()