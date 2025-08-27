from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo
from .config import Config

mongo = PyMongo()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    CORS(app)
    mongo.init_app(app)
    
    # Register blueprints
    from .routes import main, pantry, recipes, detection
    app.register_blueprint(main.bp)
    app.register_blueprint(pantry.bp, url_prefix='/api/pantry')
    app.register_blueprint(recipes.bp, url_prefix='/api/recipes')
    app.register_blueprint(detection.bp, url_prefix='/api/detection')
    
    return app
