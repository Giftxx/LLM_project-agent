import pdfplumber
import json
from pathlib import Path
from typing import Dict, List, Tuple
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import numpy as np
import cv2
import hashlib

try:
    from PIL import Image
except ImportError:
    Image = None

# Use absolute paths
BASE_DIR = Path().absolute().parent
RAW_DIR = BASE_DIR / "1_data"
PARSED_DIR = BASE_DIR / "2_parsed"
TEXT_DIR = PARSED_DIR / "text"
MANIFEST_DIR = PARSED_DIR / "manifests"
IMAGES_DIR = PARSED_DIR / "images"
FIGURES_DIR = PARSED_DIR / "figures"  # สำหรับเก็บรูปที่ดึงออกมาจาก PDF

print(f"Looking for PDF files in: {RAW_DIR}")

# Create directories if they don't exist
for dir_path in [TEXT_DIR, MANIFEST_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def compute_image_hash(img_region: np.ndarray, hash_type: str = 'structural') -> str:
    """Compute hash of image region for duplicate detection
    
    Args:
        img_region: Image region as numpy array
        hash_type: Type of hash ('average', 'perceptual', or 'structural')
    
    Returns:
        Hash string
    """
    try:
        # ใช้ Structural Hash เป็นค่าเริ่มต้น (MD5 ของไฟล์รูป)
        img_bytes = cv2.imencode('.png', img_region)[1].tobytes()
        hash_str = hashlib.md5(img_bytes).hexdigest()
        
        # ถ้ามี PIL สามารถใช้ perceptual hash
        if Image is not None and hash_type in ['average', 'perceptual']:
            if len(img_region.shape) == 3:
                img_pil = Image.fromarray(cv2.cvtColor(img_region, cv2.COLOR_BGR2RGB))
            else:
                img_pil = Image.fromarray(img_region)
            
            if hash_type == 'average':
                # Average Hash
                img_small = img_pil.convert('L').resize((8, 8))
                pixels = list(img_small.getdata())
                avg = sum(pixels) / len(pixels)
                hash_str = ''.join('1' if p > avg else '0' for p in pixels)
            elif hash_type == 'perceptual':
                # Perceptual Hash
                img_small = img_pil.convert('L').resize((32, 32))
                pixels = list(img_small.getdata())
                avg = sum(pixels) / len(pixels)
                hash_str = ''.join('1' if p > avg else '0' for p in pixels)
    except Exception as e:
        print(f"Error computing hash: {e}")
        # Fallback: ใช้ MD5 ของ numpy array
        hash_str = hashlib.md5(img_region.tobytes()).hexdigest()
    
    return hash_str

def compare_image_hashes(hash1: str, hash2: str) -> float:
    """Compare two image hashes and return similarity (0-1)
    
    Returns:
        Similarity score between 0 (different) and 1 (identical)
    """
    # ถ้าเป็น MD5 hash (ความยาว 32)
    if len(hash1) == 32 and len(hash2) == 32:
        # MD5 ตัวเดียว = identical
        return 1.0 if hash1 == hash2 else 0.0
    
    # ถ้ายาวไม่เท่า
    if len(hash1) != len(hash2):
        return 1.0 if hash1 == hash2 else 0.0
    
    # คำนวณ Hamming distance สำหรับ perceptual/average hash
    hamming = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    similarity = 1 - (hamming / len(hash1))
    return similarity

def find_duplicate_images(manifest: Dict, similarity_threshold: float = 0.95) -> List[int]:
    """Find and identify duplicate images that appear on multiple pages
    
    Args:
        manifest: The PDF manifest containing images info
        similarity_threshold: Threshold for considering images as duplicates (0-1)
    
    Returns:
        List of image indices to skip (duplicates)
    """
    if not manifest.get('images'):
        return []
    
    images = manifest['images']
    duplicates_to_skip = []
    image_hashes = {}
    
    print("Analyzing images for duplicates...")
    
    # Compute hashes for all images
    for idx, img_info in enumerate(images):
        img_path = Path(img_info['path'])
        if img_path.exists():
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    hash_val = compute_image_hash(img, 'structural')
                    image_hashes[idx] = hash_val
                else:
                    print(f"Failed to read image {img_path}")
            except Exception as e:
                print(f"Error computing hash for {img_path}: {e}")
        else:
            print(f"Image not found: {img_path}")
    
    if not image_hashes:
        print("No image hashes computed")
        return []
    
    print(f"Computed hashes for {len(image_hashes)} images")
    
    # Group images by hash value
    hash_groups = {}
    for idx, hash_val in image_hashes.items():
        if hash_val not in hash_groups:
            hash_groups[hash_val] = []
        hash_groups[hash_val].append(idx)
    
    print(f"Found {len(hash_groups)} unique image hashes")
    
    # Find hashes that appear on many pages
    total_pages = len(set(img['page'] for img in images))
    print(f"Total pages with images: {total_pages}")
    
    for hash_val, image_indices in hash_groups.items():
        # หาหน้าที่มีรูปนี้
        pages_with_this_hash = set(images[idx]['page'] for idx in image_indices)
        page_count = len(pages_with_this_hash)
        
        # ถ้ารูปนี้ปรากฏบนหลาย ๆ หน้า ให้ถือว่าเป็นรูปซ้ำ
        duplicate_threshold = max(2, int(total_pages * 0.5))  # ปรากฏบน 50% ของหน้าหรืออย่างน้อย 2 หน้า
        
        if page_count >= duplicate_threshold:
            print(f"Hash {hash_val[:16]}... appears on {page_count}/{total_pages} pages")
            print(f"  Image indices: {image_indices}")
            print(f"  Pages: {sorted(pages_with_this_hash)}")
            
            # เก็บไว้ 1 รูป ลบอันอื่น
            for idx in image_indices[1:]:  # ข้ามรูปแรก เก็บไว้
                print(f"  Marking image {idx} as duplicate")
                duplicates_to_skip.append(idx)
    
    return duplicates_to_skip

def validate_region_size(width: int, height: int, page_shape: Tuple[int, int]) -> Tuple[bool, str]:
    """Validate image region size and aspect ratio"""
    # ขนาดขั้นต่ำ (pixels)
    if width < 50 or height < 50:
        return False, f"Region too small (w={width}, h={height})"

    # สัดส่วนเทียบกับหน้า PDF
    page_height, page_width = page_shape
    width_ratio = width / page_width
    height_ratio = height / page_height
    if width_ratio < 0.03 and height_ratio < 0.03:
        return False, f"Region too small relative to page (w_ratio={width_ratio:.3f}, h_ratio={height_ratio:.3f})"

    # ตรวจสอบ aspect ratio
    aspect_ratio = width / height
    if aspect_ratio < 0.3 or aspect_ratio > 4:
        return False, f"Invalid aspect ratio ({aspect_ratio:.2f})"

    return True, ""

def is_likely_schematic(img_region: np.ndarray) -> Tuple[bool, float, float]:
    """Determine if an image region is likely to be a schematic diagram."""
    # แปลงเป็น grayscale และลดนอยส์
    gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY) if len(img_region.shape) == 3 else img_region
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # หาเส้นและมุม
    edges = cv2.Canny(blurred, 20, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=15)
    corners = cv2.goodFeaturesToTrack(gray, 150, 0.008, 8)
    
    # คำนวณความหนาแน่น
    total_pixels = gray.size
    line_density = 0 if lines is None else len(lines) / total_pixels * 10000
    corner_density = 0 if corners is None else len(corners) / total_pixels * 10000
    
    # ตรวจสอบแนวเส้นตรง
    if lines is not None and len(lines) > 0:
        angles = [abs(np.degrees(np.arctan2(line[0][3]-line[0][1], line[0][2]-line[0][0])) % 90) for line in lines]
        orthogonal_ratio = sum(1 for a in angles if a < 10 or a > 80) / len(lines)
    else:
        orthogonal_ratio = 0
    
    # เกณฑ์การตัดสินว่าเป็นแผนผัง
    is_schematic = (line_density > 0.8 and corner_density > 0.3 and
                   line_density/corner_density > 1.2 and orthogonal_ratio > 0.4)
    
    return is_schematic, line_density, corner_density

def extract_images_from_page(pdf_path: Path, page_num: int, pdf_name: str) -> List[Dict]:
    """Extract images from a PDF page using pdf2image and OpenCV
    
    Returns:
        List of dicts containing image info (path, bbox, etc.)
    """
    images_info = []
    
    try:
        print(f"Converting page {page_num} to image...")
        # แปลงหน้า PDF เป็นรูปภาพ
        poppler_path = r"C:\Program Files\poppler-25.07.0\Library\bin"
        pages = convert_from_path(
            pdf_path,
            first_page=page_num,
            last_page=page_num,
            dpi=300,
            poppler_path=poppler_path
        )
        
        if not pages:
            print(f"Failed to convert page {page_num} to image")
            return images_info
            
        # แปลงรูปภาพเป็น OpenCV format
        page_img = np.array(pages[0])
        gray = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY)
        
        # ใช้ Canny edge detection เพื่อหาขอบ
        edges = cv2.Canny(gray, 30, 100)  # ลดค่า threshold ลงเพื่อจับรายละเอียดมากขึ้น
        
        # ขยายขอบให้เชื่อมต่อกัน
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))  # ลดขนาด kernel ลง
        dilated = cv2.dilate(edges, kernel, iterations=1)  # ลดจำนวน iterations
        
        # หา contours
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        print(f"Found {len(contours)} potential image areas")
        
        # กรองและรวม contours ที่ซ้อนทับกัน
        min_area = 1400  # ลดจาก 2500 เป็น 1400 (ประมาณ 42x42 pixels) เพื่อจับตารางเล็ก
        min_overlap_ratio = 0.1  # ลดเป็น 0.1 (10%) เพื่อให้รวมแม้แต่กล่องที่มีส่วนเชื่อมต่อเพียงเล็กน้อย
        filtered_boxes = []
        
        # เรียง contours ตามขนาดจากใหญ่ไปเล็ก
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # ใช้ iterative merging เพื่อรวมกล่องที่ซ้อนทับ (ทำซ้ำจนไม่มีการเปลี่ยนแปลง)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            new_box = [x, y, x+w, y+h]
            
            # ตรวจสอบการซ้อนทับกับกล่องที่มีอยู่แล้ว
            merged = False
            for i, existing_box in enumerate(filtered_boxes):
                # คำนวณพื้นที่ที่ซ้อนทับกัน
                x_left = max(new_box[0], existing_box[0])
                y_top = max(new_box[1], existing_box[1])
                x_right = min(new_box[2], existing_box[2])
                y_bottom = min(new_box[3], existing_box[3])
                
                if x_right > x_left and y_bottom > y_top:
                    overlap_area = (x_right - x_left) * (y_bottom - y_top)
                    area1 = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
                    area2 = (existing_box[2] - existing_box[0]) * (existing_box[3] - existing_box[1])
                    min_area_box = min(area1, area2)
                    overlap_ratio = overlap_area / min_area_box if min_area_box > 0 else 0
                    
                    if overlap_ratio > min_overlap_ratio:
                        # รวมกล่องที่ซ้อนทับกัน
                        filtered_boxes[i] = [
                            min(new_box[0], existing_box[0]),
                            min(new_box[1], existing_box[1]),
                            max(new_box[2], existing_box[2]),
                            max(new_box[3], existing_box[3])
                        ]
                        merged = True
                        break
                        
            if not merged:
                filtered_boxes.append(new_box)
        
        # ทำการ merge iteratively ถ้ายังมีกล่องที่ซ้อนทับ
        changed = True
        while changed:
            changed = False
            new_filtered = []
            used = set()
            
            # พารามิเตอร์สำหรับ distance-based merging
            # ถ้าระยะห่างระหว่าง 2 กล่องน้อยกว่านี้ ให้รวมกัน
            max_distance = 50  # pixels
            
            for i, box1 in enumerate(filtered_boxes):
                if i in used:
                    continue
                    
                merged_box = list(box1)
                for j, box2 in enumerate(filtered_boxes):
                    if i >= j or j in used:
                        continue
                    
                    # ตรวจหาการซ้อนทับ
                    x_left = max(merged_box[0], box2[0])
                    y_top = max(merged_box[1], box2[1])
                    x_right = min(merged_box[2], box2[2])
                    y_bottom = min(merged_box[3], box2[3])
                    
                    should_merge = False
                    
                    # ถ้าซ้อนทับ
                    if x_right > x_left and y_bottom > y_top:
                        overlap_area = (x_right - x_left) * (y_bottom - y_top)
                        area1 = (merged_box[2] - merged_box[0]) * (merged_box[3] - merged_box[1])
                        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                        min_area_box = min(area1, area2)
                        overlap_ratio = overlap_area / min_area_box if min_area_box > 0 else 0
                        
                        if overlap_ratio > min_overlap_ratio:
                            should_merge = True
                    
                    # ถ้าอยู่ใกล้กัน (ระยะห่าง < max_distance)
                    if not should_merge:
                        # คำนวณระยะห่างระหว่างขอบของกล่อง
                        # ทั้งในแนวนอนและแนวตั้ง
                        
                        # ระยะห่างแนวนอน
                        if merged_box[2] < box2[0]:  # box1 อยู่ซ้ายของ box2
                            h_dist = box2[0] - merged_box[2]
                        elif box2[2] < merged_box[0]:  # box2 อยู่ซ้ายของ box1
                            h_dist = merged_box[0] - box2[2]
                        else:
                            h_dist = 0
                        
                        # ระยะห่างแนวตั้ง
                        if merged_box[3] < box2[1]:  # box1 อยู่บนของ box2
                            v_dist = box2[1] - merged_box[3]
                        elif box2[3] < merged_box[1]:  # box2 อยู่บนของ box1
                            v_dist = merged_box[1] - box2[3]
                        else:
                            v_dist = 0
                        
                        # ถ้าระยะห่างน้อยกว่า max_distance ในทิศทางใดทิศทางหนึ่ง
                        if (h_dist < max_distance or v_dist < max_distance) and (h_dist < max_distance and v_dist < max_distance):
                            should_merge = True
                    
                    if should_merge:
                        # รวมกล่อง
                        merged_box = [
                            min(merged_box[0], box2[0]),
                            min(merged_box[1], box2[1]),
                            max(merged_box[2], box2[2]),
                            max(merged_box[3], box2[3])
                        ]
                        used.add(j)
                        changed = True
                
                new_filtered.append(merged_box)
            
            filtered_boxes = new_filtered
        
        # บันทึกภาพจากกล่องที่ผ่านการกรอง
        for idx, box in enumerate(filtered_boxes, 1):
            x, y = box[0], box[1]
            w = box[2] - box[0]
            h = box[3] - box[1]
            
            # กรองตามขนาดจริง (pixels) - ลดลงเพื่อจับตารางเล็ก
            min_width = 60  # ความกว้างขั้นต่ำ (ลดจาก 100)
            min_height = 30  # ความสูงขั้นต่ำ (ลดจาก 100 เพื่อจับตารางเล็ก)
            
            if w < min_width or h < min_height:
                print(f"Region too small (w={w}, h={h}), skipping...")
                continue
            
            # ลบรูปที่มีพื้นที่น้อยเกินไป (logo, ลายเซ็น ฯลฯ)
            # ใช้เกณฑ์ area < 100,000 pixels
            area = w * h
            if area < 100000:
                print(f"Image too small (area={area:,} px < 100k), skipping...")
                continue
            
            # กรองตามสัดส่วนของขนาดเทียบกับหน้า PDF
            page_height, page_width = page_img.shape[:2]
            width_ratio = w / page_width
            height_ratio = h / page_height
            
            min_page_ratio = 0.01  # ลดจาก 3% เป็น 1% เพื่อจับตารางเล็ก
            if width_ratio < min_page_ratio and height_ratio < min_page_ratio:
                print(f"Region too small relative to page (w_ratio={width_ratio:.3f}, h_ratio={height_ratio:.3f}), skipping...")
                continue
            
            # ตรวจสอบอัตราส่วนของภาพ - รองรับตาราง
            aspect_ratio = w / h
            if aspect_ratio < 0.15 or aspect_ratio > 12:  # เพิ่มจาก 10 เป็น 12 สำหรับตารางที่กว้าง
                print(f"Invalid aspect ratio ({aspect_ratio:.2f}), skipping...")
                continue
            
            # ลบรูป 4 จตุรัสขนาดเล็กมาก (เช่น logo, watermark ที่เป็นสี่เหลี่ยมจัตุรัสเล็กน้อย)
            # ตรวจสอบว่าเป็นรูปสี่เหลี่ยมจัตุรัสขนาดเล็ก หรือรูป logo Kbyte
            is_small_square = (w < 100 and h < 100 and 0.8 <= aspect_ratio <= 1.2)
            
            # ตัดส่วนรูปภาพเพื่อตรวจสอบค่า
            img_region = page_img[y:y+h, x:x+w]
            
            # ลบรูปขนาดเล็กที่มีไฟล์ขนาด < 25KB หรือที่มีความเหมือนกับ logo
            avg_color = cv2.mean(img_region)[:3]
            is_likely_logo = (
                # รูปสี่เหลี่ยมจัตุรัสขนาดเล็ก
                is_small_square or
                # รูปที่มีพื้นที่น้อย < 200 pixels
                (w * h < 200) or
                # รูปที่มีความสูงเล็ก < 80 pixels และความกว้างไม่มากนัก < 600 pixels (ลายเซ็นหรือ logo ที่ยาว)
                (h < 80 and w < 600 and min(h, w) < 50)
            )
            
            if is_likely_logo:
                print(f"Logo/watermark detected (w={w}, h={h}, ratio={aspect_ratio:.2f}, area={w*h}), skipping...")
                continue
            
            # วิเคราะห์ว่าเป็นแผนผังหรือไม่
            is_schematic, line_density, corner_density = is_likely_schematic(img_region)
            
            # ตรวจสอบความซับซ้อนของภาพ
            complexity = cv2.Laplacian(cv2.cvtColor(img_region, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
            if complexity < 50 and not is_schematic:  # ลดค่า threshold สำหรับแผนผัง
                continue
            
            # สร้างชื่อไฟล์
            image_type = "schematic" if is_schematic else "image"
            image_filename = f"{pdf_name}_p{page_num:04d}_{image_type}{idx:03d}.png"
            image_path = FIGURES_DIR / image_filename
            
            # บันทึกรูป
            cv2.imwrite(str(image_path), cv2.cvtColor(img_region, cv2.COLOR_RGB2BGR))
            
            # เก็บข้อมูล
            images_info.append({
                'path': str(image_path),
                'page': page_num,
                'bbox': box,
                'method': 'opencv',
                'index': idx,
                'area': w * h,
                'aspect_ratio': aspect_ratio,
                'complexity': complexity,
                'is_schematic': is_schematic,
                'line_density': line_density,
                'corner_density': corner_density
            })
            print(f"Saved {image_type}: {image_path}")
    
    except Exception as e:
        print(f"Error processing page {page_num}: {str(e)}")
    
    return images_info

def get_or_create_manifest(pdf_path: Path) -> Dict:
    """Get existing manifest or create a new one for the PDF file"""
    manifest_file = MANIFEST_DIR / f"{pdf_path.stem}_manifest.json"
    if manifest_file.exists():
        return json.loads(manifest_file.read_text(encoding="utf-8"))
    return {
        "pdf_file": str(pdf_path.name),
        "total_pages": 0,
        "processed_pages": [],
        "processing_method": {},  # page_num -> method used (direct or ocr)
        "images": []  # เก็บข้อมูลรูปภาพที่ดึงออกมา
    }

def update_manifest(pdf_path: Path, manifest: Dict):
    """Save the updated manifest back to file"""
    manifest_file = MANIFEST_DIR / f"{pdf_path.stem}_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

def needs_ocr(page, text: str) -> bool:
    """Determine if a page needs OCR based on text content"""
    text = text.strip()
    if not text:
        print("No text found in direct extraction, will use OCR")
        return True
    return False

def render_page(pdf_path: Path, page_num: int) -> Path:
    """Render a PDF page as image for OCR processing"""
    # Set poppler path directly - make sure this path matches your installation
    poppler_path = r"C:\Program Files\poppler-25.07.0\Library\bin"
    # Convert PDF page to image (300 DPI for good OCR results)
    images = convert_from_path(
        pdf_path,
        first_page=page_num,
        last_page=page_num,
        dpi=300,
        poppler_path=poppler_path
    )
    if not images:
        raise ValueError(f"Failed to render page {page_num} from {pdf_path}")
    
    # Save the rendered image
    image_file = IMAGES_DIR / f"{pdf_path.stem}_p{page_num:04d}.png"
    images[0].save(str(image_file), 'PNG')
    return image_file

def process_with_ocr(image_path: Path) -> str:
    """Process an image with PaddleOCR"""
    print("Starting OCR processing...")
    # Initialize PaddleOCR with English language support
    ocr = PaddleOCR(
        use_textline_orientation=True,  # รองรับข้อความที่เอียง
        lang='en'                       # ใช้แค่ภาษาอังกฤษ
    )
    
    # Run OCR on the image using new API
    result = ocr.predict(str(image_path))
    if not result:
        print("OCR returned no results")
        return ""
        
    print(f"OCR found {len(result)} text regions")
    
    # Extract and combine text from all detected regions
    texts = []
    for line in result:
        if isinstance(line, list) and len(line) >= 2:
            # Get the text part from the result
            text = line[1][0] if isinstance(line[1], tuple) else line[1]
            texts.append(text)
    
    return '\n'.join(texts)

def find_related_images(pdf_name: str, page_num: int) -> List[Dict]:
    """Find images from the manifest that are on the specified page"""
    manifest_file = MANIFEST_DIR / f"{pdf_name}_manifest.json"
    if not manifest_file.exists():
        return []
        
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    
    # หารูปภาพทั้งหมดที่อยู่ในหน้าที่ระบุ
    related_images = []
    for img in manifest.get('images', []):
        if img['page'] == page_num:
            # ตรวจสอบว่าไฟล์รูปยังมีอยู่
            if Path(img['path']).exists():
                related_images.append(img)
            
    return related_images

def extract_text(pdf_path: Path, start_page: int = None, end_page: int = None):
    """Extract text from PDF pages with OCR fallback for scanned pages
    
    Args:
        pdf_path: Path to PDF file
        start_page: First page to process (1-based), if None process from start
        end_page: Last page to process (1-based), if None process until end
    """
    print("\nStarting text extraction...")
    # Get or create manifest for tracking progress
    manifest = get_or_create_manifest(pdf_path)
    
    with pdfplumber.open(pdf_path) as pdf:
        # Update total pages in manifest
        total_pages = len(pdf.pages)
        manifest["total_pages"] = total_pages
        print(f"Total pages in PDF: {total_pages}")
        
        # กำหนดช่วงหน้าที่จะประมวลผล
        start_page = start_page if start_page else 1
        end_page = end_page if end_page else total_pages
        
        # ตรวจสอบค่าที่รับเข้ามา
        if start_page < 1 or start_page > total_pages:
            raise ValueError(f"start_page must be between 1 and {total_pages}")
        if end_page < start_page or end_page > total_pages:
            raise ValueError(f"end_page must be between {start_page} and {total_pages}")
            
        print(f"Processing pages {start_page} to {end_page}")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            # ข้ามหน้าที่ไม่ได้อยู่ในช่วงที่กำหนด
            if page_num < start_page or page_num > end_page:
                continue
                
            print(f"\n{'='*50}")
            print(f"Processing page {page_num}/{total_pages}")
            
            # Skip if page already processed
            if page_num in manifest["processed_pages"]:
                print(f"Page {page_num} already processed, skipping...")
                continue
            
            print(f"Processing page {page_num}...")
            # ดึงรูปภาพก่อน
            images_info = extract_images_from_page(pdf_path, page_num, pdf_path.stem)
            manifest["images"].extend(images_info)
            
            # ดึงข้อความ
            print(f"Extracting text from page {page_num}...")
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            if not text.strip():
                print("First attempt failed, trying with different settings...")
                text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            
            print(f"Characters found in direct extraction: {len(text)}")
            
            # Check if OCR is needed
            if needs_ocr(page, text):
                print(f"Low text density detected in page {page_num}, using OCR...")
                # Render page to image
                image_path = render_page(pdf_path, page_num)
                # Process with OCR
                text = process_with_ocr(image_path)
                # Clean up the temporary image file
                image_path.unlink()
                # Update manifest with processing method
                manifest["processing_method"][str(page_num)] = "ocr"
            else:
                manifest["processing_method"][str(page_num)] = "direct"
            
            # Save extracted text
            out_file = TEXT_DIR / f"{pdf_path.stem}_p{page_num:04d}.txt"
            out_file.write_text(text, encoding="utf-8")
            print(f"Saved text: {out_file}")
            
            # Update manifest
            manifest["processed_pages"].append(page_num)
            update_manifest(pdf_path, manifest)

def search_text_with_images(query: str, search_dir: Path = TEXT_DIR):
    """ค้นหาข้อความและรูปภาพที่เกี่ยวข้อง"""
    results = []
    
    # ค้นหาในไฟล์ .txt ทั้งหมด
    for txt_file in search_dir.glob("*.txt"):
        try:
            content = txt_file.read_text(encoding='utf-8')
            if query.lower() in content.lower():
                # แยกชื่อไฟล์และหน้า
                pdf_name = txt_file.stem.rsplit('_p', 1)[0]
                page_num = int(txt_file.stem.split('_p')[-1])
                
                # หารูปภาพที่เกี่ยวข้อง
                images = find_related_images(pdf_name, page_num)
                
                results.append({
                    'pdf_name': pdf_name,
                    'page': page_num,
                    'text_file': txt_file,
                    'related_images': images
                })
        except Exception as e:
            print(f"Error processing {txt_file}: {str(e)}")
            continue
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract text and search with related images')
    parser.add_argument('--start', type=int, help='Start page number (1-based)', default=None)
    parser.add_argument('--end', type=int, help='End page number (1-based)', default=None)
    parser.add_argument('--search', type=str, help='Search text in extracted content', default=None)
    parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicate images that appear on every page')
    args = parser.parse_args()
    
    if args.remove_duplicates:
        # ลบรูปซ้ำ
        pdf_files = list(RAW_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {RAW_DIR}")
        else:
            for pdf_file in pdf_files:
                manifest_file = MANIFEST_DIR / f"{pdf_file.stem}_manifest.json"
                if manifest_file.exists():
                    print(f"\n{'='*50}")
                    print(f"Processing duplicates for: {pdf_file.name}")
                    print(f"{'='*50}")
                    
                    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
                    duplicates = find_duplicate_images(manifest, similarity_threshold=0.90)
                    
                    if duplicates:
                        print(f"\nFound {len(duplicates)} duplicate images to remove")
                        
                        # ลบไฟล์รูปที่ซ้ำ
                        removed_count = 0
                        for idx in duplicates:
                            img_info = manifest['images'][idx]
                            img_path = Path(img_info['path'])
                            if img_path.exists():
                                try:
                                    img_path.unlink()
                                    removed_count += 1
                                    print(f"Deleted: {img_path.name}")
                                except Exception as e:
                                    print(f"Error deleting {img_path}: {e}")
                            
                            # ทำเครื่องหมายว่าเป็นรูปซ้ำในข้อมูล
                            manifest['images'][idx]['is_duplicate'] = True
                        
                        # บันทึก manifest ที่อัปเดต
                        update_manifest(pdf_file, manifest)
                        print(f"\nRemoved {removed_count} duplicate images")
                        print(f"Updated manifest: {manifest_file}")
                    else:
                        print("No duplicate images found")
                else:
                    print(f"No manifest found for {pdf_file.name}")
    elif args.search:
        # ค้นหาข้อความและรูปภาพที่เกี่ยวข้อง
        results = search_text_with_images(args.search)
        if results:
            print(f"\nFound {len(results)} matches for '{args.search}':")
            for result in results:
                print(f"\nIn {result['pdf_name']}, page {result['page']}:")
                print(f"Text file: {result['text_file']}")
                if result['related_images']:
                    print("Related images:")
                    for img in result['related_images']:
                        print(f"- {img['path']} ({img['method']})")
                else:
                    print("No related images found on this page")
        else:
            print(f"\nNo matches found for '{args.search}'")
    else:
        # ประมวลผล PDF ตามปกติ
        pdf_files = list(RAW_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {RAW_DIR}")
        else:
            print(f"Found {len(pdf_files)} PDF files")
            for pdf_file in pdf_files:
                print(f"\n{'='*50}")
                print(f"Processing: {pdf_file}")
                print(f"{'='*50}")
                try:
                    extract_text(pdf_file, args.start, args.end)
                except Exception as e:
                    print(f"Error processing {pdf_file}: {str(e)}")