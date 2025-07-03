import uvicorn
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks, UploadFile, File, Query, HTTPException, Request, Response
from pydantic import BaseModel
from chromadb.config import Settings
from chromadb import Client
from pathlib import Path
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from markitdown import MarkItDown
from PyPDF2 import PdfReader
import logging
import pytesseract
from PIL import Image
import requests
import base64
import numpy as np
import datetime
from dotenv import load_dotenv
import cv2
import asyncio, tempfile, uuid, json, os, functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from zipfile import ZipFile
from bs4 import BeautifulSoup
import redis


load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Where ChromaDB should persist data
PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIRECTORY", "/app/chroma_db_data")

# Configure Chroma
settings = Settings(
    persist_directory=PERSIST_DIR,
    anonymized_telemetry=False
)

# Create a global ChromaDB client (reuse instead of creating a new one each route)
chroma_client = Client(settings)

# Initialize embedding model
embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')



# Image storage directory
IMAGES_DIR = os.path.join(os.getcwd(), "stored_images")
os.makedirs(IMAGES_DIR, exist_ok=True)


# Create a standard FastAPI app
app = FastAPI(title="ChromaDB Dockerized")

# very simple in-memory store; swap out for Redis/db in prod
# jobs: Dict[str, str] = {}
# Redis-backed job store
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

open_ai_api_key = os.getenv("OPEN_AI_API_KEY")  

VISION_CONFIG = {
    "openai_enabled": bool(open_ai_api_key),
    "ollama_enabled": True, 
    "huggingface_enabled": True,  
    "enhanced_local_enabled": True,
    "ollama_url": os.getenv("OLLAMA_URL", "http://ollama:11434"),
    "ollama_model": os.getenv("OLLAMA_VISION_MODEL", "llava"),
    "huggingface_model": os.getenv("HUGGINGFACE_VISION_MODEL", "Salesforce/blip-image-captioning-base")
}

_hf_processor = None
_hf_model = None


def get_markitdown_instance(api_key_override: str = None):
    """Create MarkItDown instance with proper configuration"""
    api_key = api_key_override or open_ai_api_key
    
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            return MarkItDown(llm_client=client, llm_model="gpt-4o-mini")
        except Exception as e:
            logger.error(f"Failed to create OpenAI MarkItDown instance: {e}")
            return MarkItDown()
    else:
        logger.info("No OpenAI API key available, using basic MarkItDown")
        return MarkItDown()


async def describe_with_openai_markitdown(image_path: str, api_key: str = open_ai_api_key) -> Optional[str]:
    """Use OpenAI via MarkItDown for image description"""
    try:
        if not (api_key or open_ai_api_key):
            logger.info("No Open AI API key available for image description")
            return None
            
        # Verify image file exists and is readable
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        # Verify it's a valid image
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            logger.error(f"Invalid image file {image_path}: {e}")
            return None
            
        md_instance = get_markitdown_instance(api_key)
        fn = functools.partial(md_instance.convert, image_path)
        
        try:
            # 60s should be plenty for a single image—tune to your needs
            result =  await asyncio.wait_for(asyncio.to_thread(fn), timeout=60)
        except asyncio.TimeoutError:
            logger.warning(f"OpenAI vision timed out for {image_path}")
            return None
        
        # Extract text content properly
        desc = None
        if hasattr(result, 'text_content'):
            desc = result.text_content
        elif hasattr(result, 'content'):
            desc = result.content
        else:
            desc = str(result)
        
        if desc and desc.strip() and len(desc.strip()) > 10:
            # Clean up the description
            desc_clean = desc.strip()
            # Remove common MarkItDown artifacts
            if desc_clean.startswith("!["):
                # If it's just markdown image syntax, extract alt text
                if "](" in desc_clean:
                    alt_text = desc_clean.split("](")[0][2:]  # Remove ![
                    if alt_text:
                        desc_clean = alt_text
            
            return f"OpenAI Vision: {desc_clean}"
        
        logger.warning(f"OpenAI MarkItDown returned empty or too short description for {image_path}")
        return None
        
    except Exception as e:
        logger.warning(f"OpenAI MarkItDown failed for {image_path}: {e}")
        return None
    
def describe_with_ollama_vision(image_path: str) -> Optional[str]:
    """Use Ollama vision model for image description"""
    try:
        if not VISION_CONFIG["ollama_enabled"]:
            return None
            
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        # Read and encode image
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        # Ollama API call
        response = requests.post(
            f"{VISION_CONFIG['ollama_url']}/api/generate",
            json={
                "model": VISION_CONFIG['ollama_model'],
                "prompt": "Describe this image succicntly, focusing on the text found in the image. Be specific and descriptive.",
                "images": [img_data],
                "stream": False
            },
            timeout= (10, 60)
        )
        
        if response.status_code == 200:
            result = response.json()
            description = result.get("response", "").strip()
            if description and len(description) > 10:
                return f"Ollama Vision ({VISION_CONFIG['ollama_model']}): {description}"
        else:
            logger.warning(f"Ollama API returned status {response.status_code}")
        
        return None
        
    except Exception as e:
        logger.warning(f"Ollama vision model failed for {image_path}: {e}")
        # Disable Ollama for subsequent attempts if it fails
        VISION_CONFIG["ollama_enabled"] = False
        return None


def describe_with_huggingface_vision(image_path: str) -> Optional[str]:
    """Use Hugging Face vision model for image description"""
    try:
        if not VISION_CONFIG["huggingface_enabled"]:
            return None
            
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        global _hf_processor, _hf_model
        
        # Load models once and cache them
        if _hf_processor is None or _hf_model is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            logger.info(f"Loading HuggingFace model: {VISION_CONFIG['huggingface_model']}")
            _hf_processor = BlipProcessor.from_pretrained(VISION_CONFIG['huggingface_model'])
            _hf_model = BlipForConditionalGeneration.from_pretrained(VISION_CONFIG['huggingface_model'])
        
        # Process image
        image = Image.open(image_path)
        inputs = _hf_processor(image, return_tensors="pt")
        
        # Generate description
        out = _hf_model.generate(**inputs, max_length=50, num_beams=4)
        description = _hf_processor.decode(out[0], skip_special_tokens=True)
        
        if description and len(description) > 5:
            return f"HuggingFace BLIP: {description}"
        
        return None
        
    except Exception as e:
        logger.warning(f"HuggingFace vision model failed for {image_path}: {e}")
        # Disable HuggingFace for subsequent attempts if it fails
        VISION_CONFIG["huggingface_enabled"] = False
        return None

def enhanced_local_image_analysis(image_path: str) -> str:
    """Enhanced local image analysis using OpenCV and PIL"""
    try:
        if not os.path.exists(image_path):
            return f"Image file not found: {Path(image_path).name}"
            
        # Load images
        img_pil = Image.open(image_path)
        img_cv = cv2.imread(image_path)
        
        # Basic info
        width, height = img_pil.size
        mode = img_pil.mode
        format_info = img_pil.format or "Unknown"
        
        description_parts = []
        
        # Size classification
        total_pixels = width * height
        if total_pixels > 2000000:  # > 2MP
            size_desc = "high-resolution"
        elif total_pixels > 500000:  # > 0.5MP
            size_desc = "medium-resolution"
        else:
            size_desc = "small"
        
        description_parts.append(f"{size_desc} {format_info.lower()} image")
        
        # Advanced color analysis
        if mode == 'RGB' and img_cv is not None:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Analyze color distribution
            hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            dominant_hue = np.argmax(hist_hue)
            
            # Color classification based on HSV
            if dominant_hue < 10 or dominant_hue > 170:
                color_desc = "with red/pink tones"
            elif 10 <= dominant_hue < 25:
                color_desc = "with orange/yellow tones"
            elif 25 <= dominant_hue < 75:
                color_desc = "with green tones"
            elif 75 <= dominant_hue < 130:
                color_desc = "with blue/cyan tones"
            else:
                color_desc = "with purple/magenta tones"
            
            # Check saturation and value
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_brightness = np.mean(hsv[:, :, 2])
            
            if avg_saturation < 50:
                color_desc += " (muted/grayscale)"
            elif avg_saturation > 150:
                color_desc += " (vibrant)"
            
            if avg_brightness < 85:
                color_desc += " and dark lighting"
            elif avg_brightness > 170:
                color_desc += " and bright lighting"
            
            description_parts.append(color_desc)
            
            # Shape and edge detection
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Contour detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 20:
                description_parts.append("containing many objects or complex details")
            elif len(contours) > 5:
                description_parts.append("containing several distinct elements")
            else:
                description_parts.append("with simple composition")
            
            # Text detection heuristic
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
            morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            text_contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_like_regions = 0
            for contour in text_contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10 and w > 10 and h > 5:
                    text_like_regions += 1
            
            if text_like_regions > 2:
                description_parts.append("likely containing text or symbols")
        
        # OCR attempt
        try:
            text = pytesseract.image_to_string(img_pil, config='--psm 11').strip()
            if text and len(text) > 2:
                # Clean and limit text
                text_clean = ' '.join(text.split())
                if len(text_clean) > 100:
                    text_desc = f"with readable text including: '{text_clean[:100]}...'"
                else:
                    text_desc = f"with readable text: '{text_clean}'"
                description_parts.append(text_desc)
        except Exception as ocr_error:
            logger.debug(f"OCR failed for {image_path}: {ocr_error}")
        
        # Combine description
        final_desc = "Enhanced analysis: " + ", ".join(description_parts) + f" ({width}x{height}px)"
        return final_desc
        
    except Exception as e:
        logger.warning(f"Enhanced local analysis failed for {image_path}: {e}")
        return basic_image_analysis(image_path)
    
def basic_image_analysis(image_path: str) -> str:
    """Basic image analysis fallback"""
    try:
        if not os.path.exists(image_path):
            return f"Image file not found: {Path(image_path).name}"
            
        img = Image.open(image_path)
        width, height = img.size
        mode = img.mode
        format_info = img.format or "Unknown"
        
        # Try OCR
        try:
            text = pytesseract.image_to_string(img).strip()
        except:
            text = ""
        
        description_parts = [
            f"Basic analysis: {format_info} image",
            f"{width}x{height} pixels",
            f"{mode} color mode"
        ]
        
        if text:
            description_parts.append(f"containing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        else:
            description_parts.append("no readable text detected")
        
        return ", ".join(description_parts)
        
    except Exception as e:
        logger.error(f"Basic image analysis failed for {image_path}: {e}")
        return f"Image file: {Path(image_path).name} (analysis failed)"

async def describe_images_for_pages(
    pages_data: List[Dict],
    api_key_override=None,
    run_all_models: bool = True,
    enabled_models: set[str] = frozenset(),
    vision_flags: dict[str,bool] = {}
) -> List[Dict]:
    for page in pages_data:
        descs = []
        for img_path in page["images"]:
            if run_all_models:
                all_desc = {}
                if "openai" in enabled_models and vision_flags["openai"]:
                    d = await describe_with_openai_markitdown(img_path, api_key_override)
                    if d:
                        all_desc["OpenAI"] = d
                if "ollama" in enabled_models and vision_flags.get("ollama", False):
                    d = describe_with_ollama_vision(img_path)
                    if d: all_desc["Ollama"] = d
                if "huggingface" in enabled_models and vision_flags.get("huggingface", False):
                    d = describe_with_huggingface_vision(img_path)
                    if d: all_desc["HuggingFace"] = d
                if "enhanced_local" in enabled_models and vision_flags.get("enhanced_local", False):
                    d = enhanced_local_image_analysis(img_path)
                    if d: all_desc["Enhanced Local"] = d

                # always include basic fallback
                all_desc["Basic Fallback"] = basic_image_analysis(img_path)
                descs.append(create_combined_description(all_desc, Path(img_path).name))

            else:
                # first‐success only among enabled & flagged
                d = None
                if "openai" in enabled_models and vision_flags["openai"]:
                    d = await describe_with_openai_markitdown(img_path, api_key_override)
                if not d and "ollama" in enabled_models and vision_flags.get("ollama", False):
                    d = describe_with_ollama_vision(img_path)
                if not d and "huggingface" in enabled_models and vision_flags.get("huggingface", False):
                    d = describe_with_huggingface_vision(img_path)
                if not d and "enhanced_local" in enabled_models and vision_flags.get("enhanced_local", False):
                    d = enhanced_local_image_analysis(img_path)
                descs.append(d or basic_image_analysis(img_path))

        page["image_descriptions"] = descs
    return pages_data



def extract_and_store_images_from_file(file_content: bytes, filename: str, temp_dir: str, doc_id: str) -> List[Dict]:
    """Fixed image extraction from PDF with better error handling"""
    pages_data = []

    try:
        temp_pdf_path = os.path.join(temp_dir, filename)
        with open(temp_pdf_path, 'wb') as f:
            f.write(file_content)

        reader = PdfReader(temp_pdf_path)
        logger.info(f"Successfully opened PDF with {len(reader.pages)} pages")

        for page_num, page in enumerate(reader.pages, 1):
            page_images = []
            
            try:
                # Get page resources
                if "/Resources" not in page:
                    logger.info(f"Page {page_num}: No resources found")
                    pages_data.append({
                        "page": page_num,
                        "images": page_images,
                        "text": None
                    })
                    continue
                    
                resources = page["/Resources"]
                
                # Check if resources is an IndirectObject and resolve it
                if hasattr(resources, 'get_object'):
                    resources = resources.get_object()
                
                # Check for XObject (images)
                if "/XObject" not in resources:
                    logger.info(f"Page {page_num}: No XObjects found")
                    pages_data.append({
                        "page": page_num,
                        "images": page_images,
                        "text": None
                    })
                    continue
                
                xobjects = resources["/XObject"]
                
                # Handle IndirectObject for XObjects
                if hasattr(xobjects, 'get_object'):
                    xobjects = xobjects.get_object()
                
                # Iterate through XObjects
                for obj_name in xobjects:
                    try:
                        xobj = xobjects[obj_name]
                        
                        # Handle IndirectObject
                        if hasattr(xobj, 'get_object'):
                            xobj = xobj.get_object()
                        
                        # Check if it's an image
                        if xobj.get("/Subtype") == "/Image":
                            try:
                                # Get image data
                                filters = xobj.get('/Filter')
                                # if it’s neither DCT (jpg) nor Flate (png), try PIL from-memory:
                                if filters not in ('/DCTDecode','/FlateDecode'):
                                    img_ext = 'png'
                                    try:
                                        from io import BytesIO
                                        im = Image.open(BytesIO(data))
                                        im = im.convert("RGB")
                                        im.save(img_storage_path, format="PNG")
                                        logger.info(f"Forced-PNG from {filters}: {img_filename}")
                                    except Exception:
                                        # fallback to writing raw bytes and hope verify() passes
                                        pass
                                data = xobj.get_data()
                                
                                # Determine file extension
                                if filters == '/DCTDecode':
                                    img_ext = 'jpg'
                                elif filters == '/FlateDecode':
                                    img_ext = 'png'
                                else:
                                    img_ext = 'png'  # Default to PNG
                                
                                # Create filename
                                img_filename = f"{doc_id}_page_{page_num}_{obj_name[1:]}.{img_ext}"
                                img_storage_path = os.path.join(IMAGES_DIR, img_filename)
                                
                                # Save image
                                with open(img_storage_path, "wb") as img_file:
                                    img_file.write(data)
                                    
                                # — normalize any “png” (and catch /Filter lists like ['/FlateDecode', ...])
                                if img_ext == 'png':
                                    try:
                                        with Image.open(img_storage_path) as im:
                                            # convert any weird bit-depths/modes into RGB
                                            if im.mode not in ("RGB", "L"):
                                                im = im.convert("RGB")
                                            # re-save so the file on disk is a bona-fide PNG
                                            im.save(img_storage_path)
                                        logger.info(f"Normalized PNG: {img_filename}")
                                    except Exception as e:
                                        logger.warning(f"Could not normalize {img_filename}: {e}")
                                
                                # Verify image was saved and is valid
                                if os.path.exists(img_storage_path) and os.path.getsize(img_storage_path) > 0:
                                    # Try to open with PIL to verify it's a valid image
                                    try:
                                        with Image.open(img_storage_path) as test_img:
                                            test_img.verify()
                                        page_images.append(img_storage_path)
                                        logger.info(f"Successfully stored and verified image: {img_filename}")
                                    except Exception as img_verify_error:
                                        logger.warning(f"Invalid image file created: {img_filename}, error: {img_verify_error}")
                                        # Remove invalid file
                                        if os.path.exists(img_storage_path):
                                            os.remove(img_storage_path)
                                else:
                                    logger.warning(f"Image file was not created or is empty: {img_filename}")
                                    
                            except Exception as e:
                                logger.error(f"Failed to extract image {obj_name} from page {page_num}: {e}")
                                
                    except Exception as e:
                        logger.error(f"Error processing XObject {obj_name} on page {page_num}: {e}")
                        
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
            
            pages_data.append({
                "page": page_num,
                "images": page_images,
                "text": None
            })
            
            logger.info(f"Page {page_num}: Found {len(page_images)} images")

    except Exception as e:
        logger.error(f"Error extracting images from {filename}: {e}")

    logger.info(f"Total images extracted from {filename}: {sum(len(p['images']) for p in pages_data)}")
    return pages_data


# def process_document_with_context(file_content: bytes, filename: str, temp_dir: str, doc_id: str, openai_api_key: str = None) -> Dict:
#     """Process document maintaining context and storing images and their descriptions using MarkItDown"""

#     file_extension = Path(filename).suffix.lower()
#     images_data = []

#     # --- Step 1: Extract images from file ---
#     if file_extension == '.pdf':
#         pages_data = extract_and_store_images_from_file(file_content, filename, temp_dir, doc_id)

#         # --- Step 2: Generate descriptions with MarkItDown ---
#         pages_data = describe_images_for_pages(pages_data, openai_api_key)

#         # --- Step 3: Flatten all image data ---
#         for page in pages_data:
#             for img_path, desc in zip(page["images"], page.get("image_descriptions", [])):
#                 images_data.append({
#                     "filename": Path(img_path).name,
#                     "storage_path": img_path,
#                     "description": desc,
#                     "position_marker": f"[IMAGE:{Path(img_path).name}]",
#                     "page": page["page"]
#                 })

#     # --- Step 4: Save the original file ---
#     temp_file_path = os.path.join(temp_dir, filename)
#     with open(temp_file_path, 'wb') as f:
#         f.write(file_content)

#     # --- Step 5: Run MarkItDown on entire document ---
#     try:
#         md_instance = get_markitdown_instance(openai_api_key)
#         result = md_instance.convert(temp_file_path)
#         content = result.text_content if hasattr(result, 'text_content') else str(result)
#         logger.info(f"MarkItDown extracted content length: {len(content)} characters")
#     except Exception as e:
#         logger.error(f"MarkItDown processing failed for {filename}: {e}")
#         if file_extension == '.txt':
#             content = file_content.decode('utf-8', errors='ignore')
#         else:
#             # If MarkItDown fails, create basic content structure
#             content = f"Document: {filename}\n\nContent could not be extracted via MarkItDown."

#     # --- Step 6: Enhanced image integration strategy ---
#     if images_data:
#         logger.info(f"Integrating {len(images_data)} images into document content")
        
#         # Strategy 1: If content is very short or empty, create a structured document
#         if len(content.strip()) < 50:
#             logger.info("Content is minimal, creating structured document with images")
#             enhanced_content = [f"Document: {filename}\n"]
            
#             # Group images by page
#             pages_with_images = {}
#             for img_data in images_data:
#                 page_num = img_data.get("page", 1)
#                 if page_num not in pages_with_images:
#                     pages_with_images[page_num] = []
#                 pages_with_images[page_num].append(img_data)
            
#             # Add each page with its images
#             for page_num in sorted(pages_with_images.keys()):
#                 enhanced_content.append(f"\n--- Page {page_num} ---\n")
#                 for img_data in pages_with_images[page_num]:
#                     enhanced_content.append(f"{img_data['position_marker']}")
#                     enhanced_content.append(f"Image Description: {img_data['description']}\n")
            
#             content = "\n".join(enhanced_content)
        
#         else:
#             # Strategy 2: Insert images into existing content
#             logger.info("Inserting images into existing content")
            
#             # Split content into sections (paragraphs or lines)
#             if '\n\n' in content:
#                 sections = content.split('\n\n')
#                 separator = '\n\n'
#             else:
#                 sections = content.split('\n')
#                 separator = '\n'
            
#             enhanced_sections = []
#             images_inserted = 0
            
#             # Insert images at strategic points
#             for i, section in enumerate(sections):
#                 enhanced_sections.append(section)
                
#                 # Insert an image every few sections if we have images left
#                 if (images_inserted < len(images_data) and 
#                     i > 0 and 
#                     (i % max(1, len(sections) // len(images_data)) == 0)):
                    
#                     img_data = images_data[images_inserted]
#                     image_section = f"\n{img_data['position_marker']}\nImage Description: {img_data['description']}\n"
#                     enhanced_sections.append(image_section)
#                     images_inserted += 1
#                     logger.info(f"Inserted image {images_inserted}: {img_data['filename']}")
            
#             # Add any remaining images at the end
#             while images_inserted < len(images_data):
#                 img_data = images_data[images_inserted]
#                 image_section = f"\n{img_data['position_marker']}\nImage Description: {img_data['description']}\n"
#                 enhanced_sections.append(image_section)
#                 images_inserted += 1
#                 logger.info(f"Added remaining image {images_inserted}: {img_data['filename']}")
            
#             content = separator.join(enhanced_sections)
        
#         logger.info(f"Final content length after image integration: {len(content)} characters")
        
#         # Log a sample of the final content for debugging
#         sample_content = content[:500] + "..." if len(content) > 500 else content
#         logger.info(f"Sample final content: {sample_content}")

#     return {
#         "content": content,
#         "images_data": images_data,
#         "file_type": file_extension
#     }

def smart_chunk_with_context(content: str, images_data: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Enhanced chunking that preserves image context and references"""
    
    chunks = []
    
    # Find all image marker positions in the content
    image_positions = {}
    for img in images_data:
        marker = img['position_marker']
        pos = content.find(marker)
        if pos != -1:
            image_positions[pos] = img
            logger.info(f"Found image marker '{marker}' at position {pos}")
        else:
            logger.warning(f"Image marker '{marker}' not found in content")
    
    logger.info(f"Found {len(image_positions)} image markers in content")
    
    # Split content into chunks
    start = 0
    chunk_index = 0
    
    while start < len(content):
        end = start + chunk_size
        chunk_text = content[start:end]
        
        # Adjust boundaries to preserve context (avoid breaking sentences/paragraphs)
        if end < len(content):
            # Try to break at sentence or paragraph boundaries
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n\n')
            last_single_newline = chunk_text.rfind('\n')
            
            # Choose the best break point
            break_point = max(last_period, last_newline, last_single_newline)
            
            if break_point > start + chunk_size // 2:  # Only use break point if it's not too early
                chunk_text = content[start:break_point + 1]
                end = break_point + 1
        
        # Find images that appear in this chunk
        chunk_images = []
        for pos, img_data in image_positions.items():
            if start <= pos < end:
                chunk_images.append(img_data)
                logger.info(f"Chunk {chunk_index}: Including image {img_data['filename']}")
        
        # Create chunk metadata
        chunk_data = {
            "content": chunk_text.strip(),
            "chunk_index": chunk_index,
            "start_position": start,
            "end_position": end,
            "images": chunk_images,
            "has_images": len(chunk_images) > 0
        }
        
        chunks.append(chunk_data)
        
        # Log chunk info
        logger.info(f"Chunk {chunk_index}: {len(chunk_text)} chars, {len(chunk_images)} images")
        
        chunk_index += 1
        start = end - overlap
        
        if start >= len(content):
            break
    
    logger.info(f"Created {len(chunks)} chunks total")
    return chunks

# def debug_content_and_images(content: str, images_data: List[Dict]):
#     """Debug helper to show content and image integration"""
#     logger.info("=== CONTENT AND IMAGE INTEGRATION DEBUG ===")
#     logger.info(f"Content length: {len(content)} characters")
#     logger.info(f"Number of images: {len(images_data)}")
    
#     for i, img_data in enumerate(images_data):
#         marker = img_data['position_marker']
#         pos = content.find(marker)
#         logger.info(f"Image {i+1}: {img_data['filename']} -> Marker: {marker} -> Position: {pos}")
    
#     # Show first 1000 characters of content
#     sample = content[:1000] + "..." if len(content) > 1000 else content
#     logger.info(f"Content sample: {sample}")
#     logger.info("=== END DEBUG ===")
    
def create_combined_description(all_descriptions: dict, filename: str) -> str:
    """Create a comprehensive description combining all vision model outputs"""
    
    # Header
    combined = f"=== Multi-Model Vision Analysis for {filename} ===\n\n"
    
    # Priority order for display
    model_priority = ["OpenAI", "Ollama", "HuggingFace", "Enhanced Local", "Basic Fallback"]
    
    # Add each model's description
    for model in model_priority:
        if model in all_descriptions:
            description = all_descriptions[model]
            combined += f"**{model} Analysis:**\n{description}\n\n"
    
    # Add any models not in priority list
    for model, description in all_descriptions.items():
        if model not in model_priority:
            combined += f"**{model} Analysis:**\n{description}\n\n"
    
    # Summary section
    combined += "**Analysis Summary:**\n"
    combined += f"- Total models used: {len(all_descriptions)}\n"
    combined += f"- Models: {', '.join(all_descriptions.keys())}\n"
    
    # Extract key insights (basic analysis)
    insights = extract_key_insights(all_descriptions)
    if insights:
        combined += f"- Key insights: {insights}\n"
    
    combined += "\n" + "="*50 + "\n"
    
    return combined


def extract_key_insights(all_descriptions: dict) -> str:
    """Extract key insights from multiple descriptions"""
    insights = []
    
    # Combine all description text
    all_text = " ".join(all_descriptions.values()).lower()
    
    # Size indicators
    if "high-resolution" in all_text:
        insights.append("High quality image")
    elif "small" in all_text:
        insights.append("Small/low resolution")
    
    return ", ".join(insights[:3])  # Limit to top 3 insights


def process_document_with_context_multi_model(file_content: bytes, 
                                            filename: str, 
                                            temp_dir: str, 
                                            doc_id: str, 
                                            openai_api_key: str = None, 
                                            run_all_models: bool = True, 
                                            selected_models: set[str] = frozenset(),
                                            vision_flags: dict[str,bool] = {}) -> Dict:
    """Process document with multi-model vision option"""
    
    file_extension = Path(filename).suffix.lower()
    images_data = []

    # Extract images from file
    if file_extension == '.pdf':
        # 1) extract images synchronously
        pages_data = extract_and_store_images_from_file(file_content, filename, temp_dir, doc_id)

        # 2) run async describer on its own loop
        loop = asyncio.new_event_loop()
        try:
            pages_data = loop.run_until_complete(
                describe_images_for_pages(
                    pages_data,
                    api_key_override=openai_api_key,
                    run_all_models=run_all_models,
                    enabled_models=selected_models,
                    vision_flags=vision_flags
                )
            )
        finally:
            loop.close()

        # 3) Flatten all image data
        for page in pages_data:
            for img_path, desc in zip(page["images"], page.get("image_descriptions", [])):
                images_data.append({
                    "filename": Path(img_path).name,
                    "storage_path": img_path,
                    "description": desc,
                    "position_marker": f"[IMAGE:{Path(img_path).name}]",
                    "page": page["page"]
                })

    # Rest of processing remains the same as your existing function...
    temp_file_path = os.path.join(temp_dir, filename)
    with open(temp_file_path, 'wb') as f:
        f.write(file_content)

    try:
        md_instance = get_markitdown_instance(openai_api_key)
        result = md_instance.convert(temp_file_path)
        content = result.text_content if hasattr(result, 'text_content') else str(result)
        logger.info(f"MarkItDown extracted content length: {len(content)} characters")
    except Exception as e:
        logger.error(f"MarkItDown processing failed for {filename}: {e}")
        if file_extension == '.txt':
            content = file_content.decode('utf-8', errors='ignore')
        else:
            content = f"Document: {filename}\n\nContent could not be extracted via MarkItDown."

    # Enhanced image integration strategy (same as before)
    if images_data:
        logger.info(f"Integrating {len(images_data)} images into document content")
        
        if len(content.strip()) < 50:
            logger.info("Content is minimal, creating structured document with images")
            enhanced_content = [f"Document: {filename}\n"]
            
            pages_with_images = {}
            for img_data in images_data:
                page_num = img_data.get("page", 1)
                if page_num not in pages_with_images:
                    pages_with_images[page_num] = []
                pages_with_images[page_num].append(img_data)
            
            for page_num in sorted(pages_with_images.keys()):
                enhanced_content.append(f"\n--- Page {page_num} ---\n")
                for img_data in pages_with_images[page_num]:
                    enhanced_content.append(f"{img_data['position_marker']}")
                    enhanced_content.append(f"Image Description: {img_data['description']}\n")
            
            content = "\n".join(enhanced_content)
        
        else:
            logger.info("Inserting images into existing content")
            
            if '\n\n' in content:
                sections = content.split('\n\n')
                separator = '\n\n'
            else:
                sections = content.split('\n')
                separator = '\n'
            
            enhanced_sections = []
            images_inserted = 0
            
            for i, section in enumerate(sections):
                enhanced_sections.append(section)
                
                if (images_inserted < len(images_data) and 
                    i > 0 and 
                    (i % max(1, len(sections) // len(images_data)) == 0)):
                    
                    img_data = images_data[images_inserted]
                    image_section = f"\n{img_data['position_marker']}\nImage Description: {img_data['description']}\n"
                    enhanced_sections.append(image_section)
                    images_inserted += 1
                    logger.info(f"Inserted image {images_inserted}: {img_data['filename']}")
            
            while images_inserted < len(images_data):
                img_data = images_data[images_inserted]
                image_section = f"\n{img_data['position_marker']}\nImage Description: {img_data['description']}\n"
                enhanced_sections.append(image_section)
                images_inserted += 1
                logger.info(f"Added remaining image {images_inserted}: {img_data['filename']}")
            
            content = separator.join(enhanced_sections)
        
        logger.info(f"Final content length after image integration: {len(content)} characters")

    return {
        "content": content,
        "images_data": images_data,
        "file_type": file_extension
    }
    
# def process_document_with_context(file_content: bytes, filename: str, temp_dir: str, doc_id: str, openai_api_key: str = None) -> Dict:
#     """Process document maintaining context and storing images and their descriptions using MarkItDown"""

#     file_extension = Path(filename).suffix.lower()
#     images_data = []

#     # --- Step 1: Extract images from file ---
#     if file_extension == '.pdf':
#         pages_data = extract_and_store_images_from_file(file_content, filename, temp_dir, doc_id)

#         # --- Step 2: Generate descriptions with MarkItDown ---
#         pages_data = describe_images_for_pages(pages_data, openai_api_key)

#         # --- Step 3: Flatten all image data ---
#         for page in pages_data:
#             for img_path, desc in zip(page["images"], page.get("image_descriptions", [])):
#                 images_data.append({
#                     "filename": Path(img_path).name,
#                     "storage_path": img_path,
#                     "description": desc,
#                     "position_marker": f"[IMAGE:{Path(img_path).name}]",
#                     "page": page["page"]
#                 })

#     # --- Step 4: Save the original file ---
#     temp_file_path = os.path.join(temp_dir, filename)
#     with open(temp_file_path, 'wb') as f:
#         f.write(file_content)

#     # --- Step 5: Run MarkItDown on entire document ---
#     try:
#         md_instance = get_markitdown_instance(openai_api_key)
#         result = md_instance.convert(temp_file_path)
#         content = result.text_content if hasattr(result, 'text_content') else str(result)
#         logger.info(f"MarkItDown extracted content length: {len(content)} characters")
#     except Exception as e:
#         logger.error(f"MarkItDown processing failed for {filename}: {e}")
#         if file_extension == '.txt':
#             content = file_content.decode('utf-8', errors='ignore')
#         else:
#             # If MarkItDown fails, create basic content structure
#             content = f"Document: {filename}\n\nContent could not be extracted via MarkItDown."

#     # --- Step 6: Enhanced image integration strategy ---
#     if images_data:
#         logger.info(f"Integrating {len(images_data)} images into document content")
        
#         # Strategy 1: If content is very short or empty, create a structured document
#         if len(content.strip()) < 50:
#             logger.info("Content is minimal, creating structured document with images")
#             enhanced_content = [f"Document: {filename}\n"]
            
#             # Group images by page
#             pages_with_images = {}
#             for img_data in images_data:
#                 page_num = img_data.get("page", 1)
#                 if page_num not in pages_with_images:
#                     pages_with_images[page_num] = []
#                 pages_with_images[page_num].append(img_data)
            
#             # Add each page with its images
#             for page_num in sorted(pages_with_images.keys()):
#                 enhanced_content.append(f"\n--- Page {page_num} ---\n")
#                 for img_data in pages_with_images[page_num]:
#                     enhanced_content.append(f"{img_data['position_marker']}")
#                     enhanced_content.append(f"Image Description: {img_data['description']}\n")
            
#             content = "\n".join(enhanced_content)
        
#         else:
#             # Strategy 2: Insert images into existing content
#             logger.info("Inserting images into existing content")
            
#             # Split content into sections (paragraphs or lines)
#             if '\n\n' in content:
#                 sections = content.split('\n\n')
#                 separator = '\n\n'
#             else:
#                 sections = content.split('\n')
#                 separator = '\n'
            
#             enhanced_sections = []
#             images_inserted = 0
            
#             # Insert images at strategic points
#             for i, section in enumerate(sections):
#                 enhanced_sections.append(section)
                
#                 # Insert an image every few sections if we have images left
#                 if (images_inserted < len(images_data) and 
#                     i > 0 and 
#                     (i % max(1, len(sections) // len(images_data)) == 0)):
                    
#                     img_data = images_data[images_inserted]
#                     image_section = f"\n{img_data['position_marker']}\nImage Description: {img_data['description']}\n"
#                     enhanced_sections.append(image_section)
#                     images_inserted += 1
#                     logger.info(f"Inserted image {images_inserted}: {img_data['filename']}")
            
#             # Add any remaining images at the end
#             while images_inserted < len(images_data):
#                 img_data = images_data[images_inserted]
#                 image_section = f"\n{img_data['position_marker']}\nImage Description: {img_data['description']}\n"
#                 enhanced_sections.append(image_section)
#                 images_inserted += 1
#                 logger.info(f"Added remaining image {images_inserted}: {img_data['filename']}")
            
#             content = separator.join(enhanced_sections)
        
#         logger.info(f"Final content length after image integration: {len(content)} characters")
        
#         # Log a sample of the final content for debugging
#         sample_content = content[:500] + "..." if len(content) > 500 else content
#         logger.info(f"Sample final content: {sample_content}")

#     return {
#         "content": content,
#         "images_data": images_data,
#         "file_type": file_extension
#     }


def run_ingest_job(
    job_id: str,
    payloads: List[Dict[str, Any]],
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    store_images: bool,
    model_name: str,
    vision_models: List[str],
    openai_api_key: Optional[str],
):
    # initialize a hash: status + zeroed counters
    progress_key = f"job:{job_id}:progress"
    # jobs[job_id] = "running"
    redis_client.set(job_id, "running")

    # initialize the hash strictly on the “progress” key
    redis_client.hset(progress_key, mapping={
        "total_chunks":     0,
        "processed_chunks": 0
    })

    # decide thread-pool size
    max_workers = min(4, os.cpu_count() or 1)

    def get_chromadb_collection():
        # each thread gets its own client/collection
        settings = Settings(persist_directory=PERSIST_DIR, anonymized_telemetry=False)
        client = Client(settings)
        return client.get_collection(name=collection_name)

    def process_one(item):
        fname = item["filename"]
        content = item["content"]
        document_id =  uuid.uuid4().hex 

        ext = Path(fname).suffix.lower()
        # 1) extract images
        with tempfile.TemporaryDirectory() as tmp_dir:
            if ext == ".pdf":
                pages_data = extract_and_store_images_from_file(content, fname, tmp_dir, fname)

            elif ext == ".docx":
                pages_data = extract_images_from_docx(content, fname, tmp_dir, fname)

            elif ext == ".xlsx":
                pages_data = extract_images_from_xlsx(content, fname, tmp_dir, fname)

            elif ext in (".html", ".htm"):
                pages_data = extract_images_from_html(content, fname, tmp_dir, fname)

            else:
                # txt, csv, pptx, etc → no images
                pages_data = [{"page": 1, "images": [], "text": None}]

            # 2) describe images
            pages_data = asyncio.new_event_loop().run_until_complete(
                describe_images_for_pages(
                    pages_data,
                    api_key_override=openai_api_key,
                    run_all_models=len(vision_models) > 1,
                    enabled_models=set(vision_models),
                    vision_flags={m: (m in vision_models) for m in vision_models},
                )
            )

            # 3) full-document processing
            doc_data = process_document_with_context_multi_model(
                file_content=content,
                filename=fname,
                temp_dir=tmp_dir,
                doc_id=fname,
                openai_api_key=openai_api_key,
                run_all_models=len(vision_models) > 1,
                selected_models=set(vision_models),
                vision_flags={m: (m in vision_models) for m in vision_models},
            )

            # 4) chunk → embed → insert
            chunks = smart_chunk_with_context(doc_data["content"],
                                          doc_data["images_data"],
                                          chunk_size, chunk_overlap)

            # bump our total_chunks counter by however many we're about to insert
            redis_client.hincrby(progress_key, "total_chunks", len(chunks))
            coll = get_chromadb_collection()
            
            for c in chunks:
                text = c["content"]
                # build metadata dict for this chunk:
                meta = {
                    "document_id": document_id,
                    "document_name": fname,
                    "file_type": doc_data["file_type"],
                    "chunk_index": c["chunk_index"],
                    "total_chunks": len(chunks),
                    "has_images": c["has_images"],
                    "image_count": len(c["images"]),
                    "start_position": c["start_position"],
                    "end_position": c["end_position"],
                    "images_stored": store_images,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                
                filenames = [img["filename"] for img in c["images"]]
                paths     = [img["storage_path"] for img in c["images"]]
                descs     = [img["description"] for img in c["images"]]

                meta["image_filenames"]     = json.dumps(filenames)
                meta["image_storage_paths"] = json.dumps(paths)
                meta["image_descriptions"]  = json.dumps(descs)
                # embed one chunk at a time
                emb = embedding_model.encode([text], 
                                            convert_to_numpy=True, 
                                            batch_size=1).tolist()
                coll.add(documents=[text],
                        embeddings=emb,
                        metadatas=[meta],
                        ids=[f"{fname}_chunk_{c['chunk_index']}"]
                )
                redis_client.hincrby(progress_key, "processed_chunks", 1)
                logger.info(f"[{job_id}] Ingested chunk {c['chunk_index']} for {fname} with {len(c['images'])} images")
        return fname

    # launch threads
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = { pool.submit(process_one, p): p["filename"] for p in payloads }
        for fut in as_completed(futures):
            fname = futures[fut]
            try:
                fut.result()
                logger.info(f"[{job_id}] Finished ingest of {fname}")
            except Exception as e:
                logger.error(f"[{job_id}] Error ingesting {fname}: {e!r}")

    # jobs[job_id] = "success"
    # redis_client.set(job_id, "success")
    redis_client.set(job_id, "success")

def extract_images_from_docx(file_content: bytes, filename: str, temp_dir: str, doc_id: str):
    docx_path = os.path.join(temp_dir, filename)
    with open(docx_path, "wb") as f:
        f.write(file_content)

    pages_data = [{"page": 1, "images": [], "text": None}]
    with ZipFile(docx_path) as z:
        for member in z.namelist():
            if member.startswith("word/media/"):
                data = z.read(member)
                ext  = Path(member).suffix
                name = f"{doc_id}_{Path(member).name}"
                out  = os.path.join(IMAGES_DIR, name)
                with open(out, "wb") as imgf:
                    imgf.write(data)
                pages_data[0]["images"].append(out)
    return pages_data

def extract_images_from_xlsx(file_content: bytes, filename: str, temp_dir: str, doc_id: str):
    xlsx_path = os.path.join(temp_dir, filename)
    with open(xlsx_path, "wb") as f:
        f.write(file_content)
    
    pages_data = [{"page": 1, "images": [], "text": None}] 
    
    with ZipFile(xlsx_path) as z:
        for m in z.namelist():
            if m.startswith("xl/media/"):
                data = z.read(m)
                ext  = Path(m).suffi
                name = f"{doc_id}_{Path(m).name}"
                out  = os.path.join(IMAGES_DIR, name)
                with open(out, "wb") as imgf:
                    imgf.write(data)
                pages_data[0]["images"].append(out)
    return pages_data

def extract_images_from_html(html_bytes, filename, temp_dir, doc_id):
    html = html_bytes.decode("utf8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    pages = [{"page":1, "images":[], "text": soup.get_text()}]
    for i, img in enumerate(soup.find_all("img"),1):
        src = img.get("src","")
        if src.startswith("data:image/"):
            header, b64 = src.split(",",1)
            ext = header.split(";")[0].split("/")[1]
            name = f"{doc_id}_{i}.{ext}"
            out  = os.path.join(IMAGES_DIR, name)
            with open(out,"wb") as f: f.write(base64.b64decode(b64))
        elif src.startswith("http"):
            resp = requests.get(src, timeout=10)
            ext  = Path(src).suffix or ".jpg"
            name = f"{doc_id}_{i}{ext}"
            out  = os.path.join(IMAGES_DIR, name)
            with open(out,"wb") as f: f.write(resp.content)
        else:
            continue
        pages[0]["images"].append(out)
    return pages

### -------------------------------------------------------------------------- ###
### Collection Endpoints ###
### -------------------------------------------------------------------------- ###


### Health Checks ###

@app.get("/")
def root_health_check():
    """Basic health check."""
    return {"status": "ok", "detail": "ChromaDB custom server running."}

@app.get("/health")
def health_check():
    """Enhanced health check endpoint with vision model status"""
    return {
        "status": "ok",
        "markitdown_available": True,
        "supported_formats": ["pdf", "docx", "xlsx", "csv", "txt", "pptx", "html"],
        "embedding_model": "multi-qa-mpnet-base-dot-v1",
        "images_directory": IMAGES_DIR,
        "vision_models": {
            "openai_enabled": VISION_CONFIG["openai_enabled"],
            "ollama_enabled": VISION_CONFIG["ollama_enabled"],
            "huggingface_enabled": VISION_CONFIG["huggingface_enabled"],
            "enhanced_local_enabled": VISION_CONFIG["enhanced_local_enabled"],
        },
        "api_keys": {
            "openai_configured": bool(open_ai_api_key)
        }
    }






@app.get("/collections")
def list_collections():
    try:
        collection_names = chroma_client.list_collections()  # Now returns just names
        return {"collections": collection_names}
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to list collections")


@app.post("/collection/create")
def create_collection(collection_name: str = Query(...)):
    """
    Create a ChromaDB collection with the given name.
    """
    try:
        # Get existing collection names safely
        existing_names = chroma_client.list_collections()
        
        if collection_name in existing_names:
            raise HTTPException(
                status_code=400,
                detail=f"Collection '{collection_name}' already exists."
            )
        
        chroma_client.create_collection(collection_name)
        logger.info(f"Created collection: {collection_name}")
        return {"created": collection_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")


@app.get("/collection")
def get_collection_info(collection_name: str = Query(...)):
    """
    Get basic info about a single collection.
    """
    try:
        # Check if collection exists
        existing_names = chroma_client.list_collections()
                
        if collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found."
            )
        
        collection = chroma_client.get_collection(name=collection_name)
        return {"name": collection.name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")


@app.delete("/collection")
def delete_collection(collection_name: str = Query(...)):
    """
    Delete a ChromaDB collection by name.
    """
    try:
        # Get existing collection names safely
        existing_collections = chroma_client.list_collections()


        if collection_name not in existing_collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found."
            )

        chroma_client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
        return {"deleted": collection_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


@app.put("/collection/edit")
def edit_collection_name(old_name: str = Query(...), new_name: str = Query(...)):
    """
    Rename a ChromaDB collection from 'old_name' to 'new_name'.
    """
    try:
        # Get existing collection names
        existing_names = chroma_client.list_collections()

        if old_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{old_name}' not found."
            )

        if new_name in existing_names:
            raise HTTPException(
                status_code=400,
                detail=f"Collection '{new_name}' already exists. Choose a different name."
            )

        # Retrieve the old collection
        collection = chroma_client.get_collection(name=old_name)

        # Create a new collection with the new name
        new_collection = chroma_client.create_collection(name=new_name)

        # Retrieve all documents from the old collection
        old_docs = collection.get()
        if old_docs and "ids" in old_docs and "documents" in old_docs:
            new_collection.add(ids=old_docs["ids"], documents=old_docs["documents"])

        # Delete the old collection
        chroma_client.delete_collection(old_name)

        return {"old_name": old_name, "new_name": new_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error editing collection: {str(e)}")


### Document Endpoints ###
class DocumentAddRequest(BaseModel):
    collection_name: str
    documents: list[str]
    ids: list[str]
    embeddings: list[list[float]] = None  
    metadatas: list[dict] = None           

@app.post("/documents/add")
def add_documents(req: DocumentAddRequest):
    try:
        # Check if the collection exists
        existing_names = chroma_client.list_collections()
                
        if req.collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{req.collection_name}' not found."
            )
        
        # Retrieve the collection
        collection = chroma_client.get_collection(req.collection_name)
        
        # Add documents along with embeddings and metadatas
        collection.add(
            documents=req.documents,
            ids=req.ids,
            embeddings=req.embeddings,
            metadatas=req.metadatas
        )
        return {
            "collection": req.collection_name,
            "added_count": len(req.documents),
            "ids": req.ids
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")


class DocumentRemoveRequest(BaseModel):
    collection_name: str
    ids: list[str]

@app.post("/documents/remove")
def remove_documents(req: DocumentRemoveRequest):
    """
    Remove documents by ID from a given collection.
    """
    try:
        # Check if the collection exists first
        existing_names = chroma_client.list_collections()
                
        if req.collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{req.collection_name}' not found."
            )

        # Now, safely retrieve the collection (since we verified it exists)
        collection = chroma_client.get_collection(req.collection_name)

        # Ensure at least one of the documents exists before attempting to delete
        existing_docs = collection.get()
        existing_ids = set(existing_docs.get("ids", []))

        if not any(doc_id in existing_ids for doc_id in req.ids):
            raise HTTPException(
                status_code=404,
                detail=f"None of the provided document IDs {req.ids} exist in collection '{req.collection_name}'."
            )

        # Delete the specified document(s)
        collection.delete(ids=req.ids)
        
        return {
            "collection": req.collection_name,
            "removed_ids": req.ids
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error removing documents: {str(e)}")


class DocumentEditRequest(BaseModel):
    collection_name: str
    doc_id: str
    new_document: str

@app.post("/documents/edit")
def edit_document(req: DocumentEditRequest):
    """
    Replace the content of an existing document by ID.
    """
    try:
        # Check if the collection exists first
        existing_names = chroma_client.list_collections()
                
        if req.collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{req.collection_name}' not found."
            )

        # Now, safely retrieve the collection
        collection = chroma_client.get_collection(req.collection_name)

        # Ensure the document exists before attempting to update
        existing_docs = collection.get()
        if req.doc_id not in existing_docs.get("ids", []):
            raise HTTPException(
                status_code=404,
                detail=f"Document '{req.doc_id}' not found in collection '{req.collection_name}'."
            )

        # Delete the old document and re-add with new content
        collection.delete(ids=[req.doc_id])
        collection.add(documents=[req.new_document], ids=[req.doc_id])

        return {
            "collection": req.collection_name,
            "updated_id": req.doc_id,
            "new_document": req.new_document
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error editing document: {str(e)}")


@app.get("/documents")
def list_documents(collection_name: str = Query(...)):
    """
    Get all documents (and their IDs) in a collection.
    """
    try:
        # Check if the collection exists first
        existing_names = chroma_client.list_collections()
                
        if collection_name not in existing_names:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")

        # Now, safely retrieve the collection
        collection = chroma_client.get_collection(name=collection_name)

        # Retrieve documents
        docs = collection.get()
        return docs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


class DocumentQueryRequest(BaseModel):
    collection_name: str
    query_embeddings: list[list[float]]
    n_results: int = 5
    include: list[str] = ["documents", "metadatas", "distances"]

@app.post("/documents/query")
def query_documents(req: DocumentQueryRequest):
    try:
        # Check if the collection exists first
        existing_names = chroma_client.list_collections()
                
        if req.collection_name not in existing_names:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{req.collection_name}' not found."
            )

        # Retrieve the collection
        collection = chroma_client.get_collection(req.collection_name)

        # Perform the query using the provided embeddings and parameters
        query_result = collection.query(
            query_embeddings=req.query_embeddings,
            n_results=req.n_results,
            include=req.include
        )
        return query_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")


@app.post("/documents/upload-and-process")
async def upload_and_process_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    collection_name: str = Query(...),
    chunk_size: int = Query(1000),
    chunk_overlap: int = Query(200),
    store_images: bool = Query(True),
    model_name: str = Query("none"),
    vision_models: str = Query(""),
    request: Request = None,
):
    # 1) Validate collection
    if collection_name not in chroma_client.list_collections():
        raise HTTPException(404, f"Collection '{collection_name}' not found")

    # 2) Generate a single job_id
    job_id = uuid.uuid4().hex
    # jobs[job_id] = "pending"
    redis_client.set(job_id, "pending")

    # 3) Slurp each UploadFile into memory so we can hand off raw bytes
    payloads: List[Dict[str,Any]] = []
    for f in files:
        content = await f.read()
        payloads.append({"filename": f.filename, "content": content})

    # 4) Kick off the background task
    selected_models = [m.strip() for m in vision_models.split(",") if m.strip()]
    background_tasks.add_task(
        run_ingest_job,
        job_id,
        payloads,
        collection_name,
        chunk_size,
        chunk_overlap,
        store_images,
        model_name,
        selected_models,
        request.headers.get("X-OpenAI-API-Key") or open_ai_api_key,
    )

    # 5) Return immediately with the job ID
    return {"job_id": job_id}



@app.get("/documents/reconstruct/{document_id}")
def reconstruct_document(document_id: str, collection_name: str = Query(...)):
    """
    Reconstruct original document from stored chunks and images with enhanced formatting.
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)

        # Get all chunks for this document
        results = collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )

        if not results["ids"]:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        # Sort chunks by chunk index
        chunks_data = []
        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            chunks_data.append({
                "chunk_index": metadata.get("chunk_index", 0),
                "content": results["documents"][i],
                "metadata": metadata
            })

        chunks_data.sort(key=lambda x: x["chunk_index"])

        document_name = chunks_data[0]["metadata"].get("document_name", "UNKNOWN")
        reconstructed_content = f"# Document: {document_name}\n\n"
        all_images = []
        image_counter = 1

        for chunk in chunks_data:
            content = chunk["content"]

            page_info = chunk["metadata"].get("page", None)
            if page_info:
                content = f"\nPage: {page_info}\n\n" + content

            # Replace image markers with markdown-style descriptions
            if chunk["metadata"].get("has_images"):
                try:
                    image_filenames = json.loads(chunk["metadata"].get("image_filenames", "[]"))
                    image_paths = json.loads(chunk["metadata"].get("image_storage_paths", "[]"))
                    image_descriptions = json.loads(chunk["metadata"].get("image_descriptions", "[]"))

                    for filename, path, desc in zip(image_filenames, image_paths, image_descriptions):
                        markdown_img = (
                            f"\n\n[Image {image_counter}]:\n"
                            f"# Description:\n"
                            f"{desc.strip()}\n"
                        )
                        marker = f"[IMAGE:{filename}]"
                        content = content.replace(marker, markdown_img)

                        all_images.append({
                            "filename": filename,
                            "storage_path": path,
                            "description": desc,
                            "exists": os.path.exists(path)
                        })
                        image_counter += 1

                except Exception as e:
                    logger.error(f"Failed to insert images: {e}")

            reconstructed_content += content + "\n\n"

        return {
            "document_id": document_id,
            "document_name": chunks_data[0]["metadata"].get("document_name", "Unknown"),
            "total_chunks": len(chunks_data),
            "reconstructed_content": reconstructed_content.strip(),
            "images": all_images,
            "metadata": {
                "file_type": chunks_data[0]["metadata"].get("file_type"),
                "total_images": len(all_images),
                "processing_timestamp": chunks_data[0]["metadata"].get("timestamp"),
                "openai_api_used": chunks_data[0]["metadata"].get("openai_api_used", False)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reconstructing document: {str(e)}")


@app.get("/images/{image_filename}")
def get_stored_image(image_filename: str):
    """
    Retrieve a stored image file.
    """
    image_path = os.path.join(IMAGES_DIR, image_filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Determine content type
        ext = Path(image_filename).suffix.lower()
        content_type = "image/jpeg" if ext == ".jpg" else "image/png"
        
        return Response(content=image_data, media_type=content_type)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image: {str(e)}")
    
@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    # status = jobs.get(job_id)
    # if status is None:
    #     raise HTTPException(404, f"Job {job_id} not found")
    # return {"job_id": job_id, "status": status}
    
    status = redis_client.get(job_id)
    prog   = redis_client.hgetall(f"job:{job_id}:progress") or {}
    return {
        "job_id": job_id,
        "status": status,
        "total_chunks":     int(prog.get("total_chunks",     0)),
        "processed_chunks": int(prog.get("processed_chunks", 0)),
    }

### Run with Uvicorn if called directly ###
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)