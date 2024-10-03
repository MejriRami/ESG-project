import os
import torch
from transformers import CLIPProcessor, CLIPModel
from llama_parser import LlamaParser
from PIL import Image

# Function to load and parse PDFs (text and images)
def load_and_parse_pdfs_with_images(pdf_folder):
    parser = LlamaParser()
    parsed_texts = []
    image_paths = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            # Parse the PDF content
            parsed_content, extracted_images = parser.parse_with_images(os.path.join(pdf_folder, filename))
            parsed_texts.append(parsed_content)
            
            # Store the extracted images for CLIP processing
            image_paths.extend(extracted_images)
    
    return parsed_texts, image_paths

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to create CLIP embeddings for text and images
def create_clip_embeddings(texts, image_paths):
    # Create embeddings for text
    text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True)
    text_embeddings = clip_model.get_text_features(**text_inputs)

    # Create embeddings for images
    image_embeddings = []
    for img_path in image_paths:
        image = Image.open(img_path)
        image_inputs = clip_processor(images=image, return_tensors="pt")
        image_embedding = clip_model.get_image_features(**image_inputs)
        image_embeddings.append(image_embedding)
    
    # Combine both text and image embeddings
    all_embeddings = torch.cat([text_embeddings] + image_embeddings, dim=0)

    return all_embeddings

# Function to store embeddings in Chroma vector store
def store_embeddings_in_chroma(embeddings, metadata):
    client = chromadb.Client()
    collection = client.create_collection("esg_reports_with_images")
    
    for emb, meta in zip(embeddings, metadata):
        collection.add(documents=[meta['text']], 
                       embeddings=[emb.cpu().numpy()], 
                       metadatas=[meta])

# Main execution
pdf_folder = 'path/to/your/pdf/folder'  # Replace with your actual path
parsed_pdf_texts, image_paths = load_and_parse_pdfs_with_images(pdf_folder)

# Create CLIP embeddings for both text and images
clip_embeddings = create_clip_embeddings(parsed_pdf_texts, image_paths)

# Store embeddings and metadata
metadata = [{'text': text} for text in parsed_pdf_texts]  # Modify with actual metadata
store_embeddings_in_chroma(clip_embeddings, metadata)
