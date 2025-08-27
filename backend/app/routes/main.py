from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
from .. import mongo
from ..models import UserInteraction

bp = Blueprint('main', __name__)

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'AI Food Waste Reducer API'
    })

@bp.route('/dashboard', methods=['GET'])
def get_dashboard():
    """Get dashboard statistics"""
    try:
        # Get pantry items count
        pantry_count = mongo.db.pantry_items.count_documents({})
        
        # Get expiring items (within 7 days)
        expiring_soon = mongo.db.pantry_items.count_documents({
            'expiry_date': {
                '$gte': datetime.now().date().isoformat(),
                '$lte': (datetime.now().date() + timedelta(days=7)).isoformat()
            }
        })
        
        # Get total waste reduced (estimated)
        waste_reduced = mongo.db.user_interactions.count_documents({
            'action': 'waste_reduced'
        }) * 0.5  # Estimate 0.5kg per interaction
        
        # Get money saved (estimated $5 per saved item)
        money_saved = waste_reduced * 10
        
        # Get recent interactions
        recent_interactions = list(mongo.db.user_interactions.find().sort('created_at', -1).limit(5))
        for interaction in recent_interactions:
            interaction['_id'] = str(interaction['_id'])
            interaction['created_at'] = interaction['created_at'].isoformat()
        
        return jsonify({
            'pantry_items': pantry_count,
            'expiring_soon': expiring_soon,
            'waste_reduced_kg': round(waste_reduced, 1),
            'money_saved_usd': round(money_saved, 2),
            'recent_interactions': recent_interactions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/interaction', methods=['POST'])
def log_interaction():
    """Log user interaction"""
    try:
        data = request.get_json()
        interaction = UserInteraction(
            action=data.get('action'),
            item_id=data.get('item_id'),
            recipe_id=data.get('recipe_id'),
            metadata=data.get('metadata', {})
        )
        
        mongo.db.user_interactions.insert_one(interaction.to_dict())
        return jsonify({'status': 'logged'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
