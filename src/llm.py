import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from loguru import logger
from src.config import get_settings
from langchain_openai import ChatOpenAI
from llama_cpp import Llama

settings = get_settings()
from huggingface_hub import hf_hub_download

def _build_llamacpp():
    if settings.gguf_model_path:
        model_path = str(settings.gguf_model_path)
    else:
        raise ValueError(
            "gguf_model_path is required when llm_provider='llamacpp'"
        )
    return Llama(
        model_path=model_path,
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=False,
    )

def _build_hf_local():
    logger.info(f"Loading local model: {settings.hf_model}...")
    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        settings.hf_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    text_gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=settings.hf_max_new_tokens,
        do_sample=settings.llm_temperature > 0,
        temperature=settings.llm_temperature,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return ChatHuggingFace(llm=llm)

def _build_gemini():
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.llm_temperature,
    )

def _build_vllm():
    return ChatOpenAI(
        model=settings.hf_model,
        api_key=settings.vllm_api_key,
        base_url=settings.vllm_api_base,
        temperature=settings.llm_temperature,
    )
    
def get_llm():
    provider = settings.llm_provider
    if provider == "hf_local":
        return _build_hf_local()
    elif provider == "gemini":
        return _build_gemini()
    elif provider == "vllm":
        return _build_vllm()
    elif provider == "llamacpp":
        return _build_llamacpp()
    else:
        raise ValueError(f"Provider {provider} is not supported.")



def invoke_llm(prompt, image_paths=None):
    if settings.llm_provider == "llamacpp":
        llm = get_llm()

        output = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return output["choices"][0]["message"]["content"]
    
    content = [{"type": "text", "text": prompt}]
    if image_paths:
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"file://{path}"
                    }
                }
            )
    message = HumanMessage(content=content)
    response = get_llm().invoke([message])
    return response.content if isinstance(response.content, str) else str(response.content)
