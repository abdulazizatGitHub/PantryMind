from flask import Blueprint, jsonify, request
from datetime import datetime, date, timedelta
from .. import mongo
from ..models import PantryItem
from bson import ObjectId

bp = Blueprint('pantry', __name__)

@bp.route('/items', methods=['GET'])
def get_pantry_items():
    """Get all pantry items"""
    try:
        items = list(mongo.db.pantry_items.find().sort('created_at', -1))
        
        # Convert ObjectId to string and add expiry status
        for item in items:
            item['_id'] = str(item['_id'])
            
            # Add expiry status
            if item.get('expiry_date'):
                try:
                    expiry_date = datetime.fromisoformat(item['expiry_date']).date()
                    days_until_expiry = (expiry_date - date.today()).days
                    
                    if days_until_expiry < 0:
                        item['expiry_status'] = 'expired'
                    elif days_until_expiry <= 3:
                        item['expiry_status'] = 'critical'
                    elif days_until_expiry <= 7:
                        item['expiry_status'] = 'warning'
                    else:
                        item['expiry_status'] = 'good'
                    
                    item['days_until_expiry'] = days_until_expiry
                except:
                    item['expiry_status'] = 'unknown'
            else:
                item['expiry_status'] = 'unknown'
        
        return jsonify({'items': items})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/items', methods=['POST'])
def add_pantry_item():
    """Add a new pantry item"""
    try:
        data = request.get_json()
        
        # Create pantry item
        item = PantryItem(
            name=data['name'],
            quantity=data.get('quantity', 1.0),
            unit=data.get('unit', 'unit'),
            expiry_date=data.get('expiry_date'),
            image_url=data.get('image_url'),
            confidence=data.get('confidence'),
            category=data.get('category')
        )
        
        # Insert into database
        result = mongo.db.pantry_items.insert_one(item.to_dict())
        
        # Log interaction
        from ..models import UserInteraction
        interaction = UserInteraction(
            action='add_item',
            item_id=str(result.inserted_id),
            metadata={'item_name': item.name}
        )
        mongo.db.user_interactions.insert_one(interaction.to_dict())
        
        return jsonify({
            'id': str(result.inserted_id),
            'message': 'Item added successfully'
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/items/<item_id>', methods=['PUT'])
def update_pantry_item(item_id):
    """Update a pantry item"""
    try:
        data = request.get_json()
        
        # Update fields
        update_data = {}
        if 'name' in data:
            update_data['name'] = data['name']
        if 'quantity' in data:
            update_data['quantity'] = data['quantity']
        if 'unit' in data:
            update_data['unit'] = data['unit']
        if 'expiry_date' in data:
            update_data['expiry_date'] = data['expiry_date']
        if 'category' in data:
            update_data['category'] = data['category']
        
        update_data['updated_at'] = datetime.utcnow()
        
        result = mongo.db.pantry_items.update_one(
            {'_id': ObjectId(item_id)},
            {'$set': update_data}
        )
        
        if result.matched_count == 0:
            return jsonify({'error': 'Item not found'}), 404
        
        return jsonify({'message': 'Item updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/items/<item_id>', methods=['DELETE'])
def delete_pantry_item(item_id):
    """Delete a pantry item"""
    try:
        result = mongo.db.pantry_items.delete_one({'_id': ObjectId(item_id)})
        
        if result.deleted_count == 0:
            return jsonify({'error': 'Item not found'}), 404
        
        # Log interaction
        from ..models import UserInteraction
        interaction = UserInteraction(
            action='delete_item',
            item_id=item_id
        )
        mongo.db.user_interactions.insert_one(interaction.to_dict())
        
        return jsonify({'message': 'Item deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/items/expiring', methods=['GET'])
def get_expiring_items():
    """Get items expiring soon"""
    try:
        # Get items expiring within 7 days
        expiring_date = (date.today() + timedelta(days=7)).isoformat()
        
        items = list(mongo.db.pantry_items.find({
            'expiry_date': {
                '$lte': expiring_date,
                '$gte': date.today().isoformat()
            }
        }).sort('expiry_date', 1))
        
        # Convert ObjectId to string
        for item in items:
            item['_id'] = str(item['_id'])
        
        return jsonify({'expiring_items': items})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
