from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
import os
import base64
from datetime import datetime
from .. import mongo
from ..ai_models import FoodDetectionModel, ExpiryDateOCR, ExpiryPredictor
from ..models import PantryItem, UserInteraction
from ..config import Config

bp = Blueprint('detection', __name__)

# Initialize AI models
food_detector = FoodDetectionModel(Config.YOLO_MODEL_PATH)
ocr_reader = ExpiryDateOCR()
expiry_predictor = ExpiryPredictor()

@bp.route('/scan', methods=['POST'])
def scan_food_item():
    """Scan image for food items and extract expiry dates"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Detect food items
        detections = food_detector.detect_food_items(image_bytes)
        
        # Extract expiry date from image
        expiry_date, ocr_text = ocr_reader.extract_expiry_date(image_bytes)
        
        # Save image to disk (optional)
        filename = secure_filename(f"scan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg")
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # Process detections
        results = []
        for detection in detections:
            # Predict expiry if not found by OCR
            predicted_expiry = None
            if not expiry_date:
                predicted_expiry = expiry_predictor.predict_expiry(
                    detection['name'], 
                    detection['category']
                )
            
            result = {
                'name': detection['name'],
                'confidence': detection['confidence'],
                'category': detection['category'],
                'bbox': detection['bbox'],
                'expiry_date': expiry_date.isoformat() if expiry_date else None,
                'predicted_expiry': predicted_expiry.isoformat() if predicted_expiry else None,
                'ocr_text': ocr_text,
                'image_url': f"/uploads/{filename}"
            }
            results.append(result)
        
        # Log interaction
        interaction = UserInteraction(
            action='scan_item',
            metadata={
                'detections_count': len(detections),
                'expiry_found': expiry_date is not None,
                'image_filename': filename
            }
        )
        mongo.db.user_interactions.insert_one(interaction.to_dict())
        
        return jsonify({
            'detections': results,
            'image_url': f"/uploads/{filename}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/scan-and-add', methods=['POST'])
def scan_and_add_item():
    """Scan image and automatically add detected items to pantry"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Detect food items
        detections = food_detector.detect_food_items(image_bytes)
        
        # Extract expiry date
        expiry_date, ocr_text = ocr_reader.extract_expiry_date(image_bytes)
        
        # Save image
        filename = secure_filename(f"scan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg")
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        added_items = []
        
        # Add each detected item to pantry
        for detection in detections:
            # Use predicted expiry if OCR didn't find one
            final_expiry = expiry_date
            if not final_expiry:
                final_expiry = expiry_predictor.predict_expiry(
                    detection['name'], 
                    detection['category']
                )
            
            # Create pantry item
            item = PantryItem(
                name=detection['name'],
                quantity=1.0,
                unit='unit',
                expiry_date=final_expiry,
                image_url=f"/uploads/{filename}",
                confidence=detection['confidence'],
                category=detection['category']
            )
            
            # Insert into database
            result = mongo.db.pantry_items.insert_one(item.to_dict())
            
            added_items.append({
                'id': str(result.inserted_id),
                'name': item.name,
                'expiry_date': final_expiry.isoformat() if final_expiry else None,
                'confidence': item.confidence
            })
        
        # Log interaction
        interaction = UserInteraction(
            action='scan_and_add',
            metadata={
                'items_added': len(added_items),
                'image_filename': filename
            }
        )
        mongo.db.user_interactions.insert_one(interaction.to_dict())
        
        return jsonify({
            'added_items': added_items,
            'message': f'Added {len(added_items)} items to pantry'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/ocr', methods=['POST'])
def extract_text():
    """Extract text from image using OCR"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Extract text and expiry date
        expiry_date, ocr_text = ocr_reader.extract_expiry_date(image_bytes)
        
        return jsonify({
            'text': ocr_text,
            'expiry_date': expiry_date.isoformat() if expiry_date else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
