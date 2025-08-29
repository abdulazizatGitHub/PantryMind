import React, { useState, useRef, useCallback } from 'react';
import './Scanner.css';

const Scanner = ({ onRecipeSelect, onAnalyticsUpdate }) => {
  const [currentStep, setCurrentStep] = useState('upload'); // upload, ingredients, recipes
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectedIngredients, setDetectedIngredients] = useState([]);
  const [generatedRecipes, setGeneratedRecipes] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  // Function to group identical ingredients and calculate average confidence
  const groupIngredients = (ingredients) => {
    const grouped = {};
    
    ingredients.forEach(item => {
      const name = item.name.toLowerCase().trim();
      if (!grouped[name]) {
        grouped[name] = {
          name: item.name,
          count: 0,
          totalConfidence: 0,
          maxConfidence: 0,
          instances: []
        };
      }
      
      grouped[name].count += 1;
      grouped[name].totalConfidence += item.confidence;
      grouped[name].maxConfidence = Math.max(grouped[name].maxConfidence, item.confidence);
      grouped[name].instances.push(item);
    });
    
    // Convert to array and calculate average confidence
    return Object.values(grouped).map(group => ({
      name: group.name,
      count: group.count,
      confidence: group.totalConfidence / group.count, // Average confidence
      maxConfidence: group.maxConfidence,
      instances: group.instances
    }));
  };

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera. Please check permissions.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  }, []);

  const captureImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      await processImage(blob);
    }, 'image/jpeg', 0.8);
  }, []);

  const processImage = async (blob) => {
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('image', blob, 'scan.jpg');

    try {
      const response = await fetch('http://localhost:8000/api/scan-and-generate-recipes', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setDetectedIngredients(result.detected_ingredients || []);
        setGeneratedRecipes(result.recipes || []);
        setAnalytics(result.analytics);
        
        // Update dashboard analytics with new data
        if (onAnalyticsUpdate && result.analytics) {
          await onAnalyticsUpdate(result.analytics);
        }
        
        setCurrentStep('ingredients');
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process image');
      }
    } catch (error) {
      console.error('Error uploading image:', error);
      alert('Error processing image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    await processImage(file);
  };

  const handleGenerateRecipes = () => {
    setCurrentStep('recipes');
  };

  const handleViewRecipes = () => {
    if (onRecipeSelect) {
      onRecipeSelect(generatedRecipes); // Pass the entire list of recipes
    }
  };

  const getIngredientIcon = (ingredientName) => {
    const name = ingredientName.toLowerCase();
    if (name.includes('apple')) return '🍎';
    if (name.includes('banana')) return '🍌';
    if (name.includes('orange')) return '🍊';
    if (name.includes('tomato')) return '🍅';
    if (name.includes('carrot')) return '🥕';
    if (name.includes('broccoli')) return '🥦';
    if (name.includes('milk')) return '🥛';
    if (name.includes('bread')) return '🍞';
    if (name.includes('chicken')) return '🍗';
    if (name.includes('fish')) return '🐟';
    if (name.includes('rice')) return '🍚';
    if (name.includes('pasta')) return '🍝';
    return '🥫';
  };

  // Upload Step
  if (currentStep === 'upload') {
    return (
      <div className="scanner-container">
        {/* Header Section */}
        <div className="scanner-header">
          <div className="scanner-header-content">
            <h1 className="scanner-title">Scan Your Ingredients</h1>
            <p className="scanner-subtitle">
              Upload an image or use your camera to detect ingredients and generate personalized recipes
            </p>
          </div>
        </div>

        {/* Main Content */}
        <div className="scanner-content">
          {/* Camera Section */}
          {/* <div className="scanner-section">
            <div className="scanner-card">
              <h2 className="section-title">
                <span className="section-icon">📷</span>
                Camera Scanner
              </h2>
              
              <div className="camera-container">
                {!videoRef.current?.srcObject ? (
                  <div className="camera-overlay">
                    <div className="camera-placeholder">
                      <div className="camera-placeholder-icon">📷</div>
                      <p>Camera not active</p>
                    </div>
                  </div>
                ) : (
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="camera-video"
                  />
                )}
                <canvas ref={canvasRef} className="hidden" />
                
                <div className="camera-controls">
                  <button
                    onClick={startCamera}
                    className="camera-button"
                  >
                    📷 Start Camera
                  </button>
                  <button
                    onClick={captureImage}
                    disabled={isProcessing || !videoRef.current?.srcObject}
                    className="camera-button capture"
                  >
                    {isProcessing ? 'Processing...' : '📸 Scan & Detect'}
                  </button>
                </div>
              </div>
            </div>
          </div> */}

          {/* Upload Section */}
          <div className="scanner-section">
            <div className="scanner-card">
              <h2 className="section-title">
                <span className="section-icon">📁</span>
                Upload Image
              </h2>
              
              <div className="upload-area">
                <input
                  type="file"
                  id="file-upload"
                  accept="image/*"
                  onChange={handleFileUpload}
                  disabled={isProcessing}
                  className="hidden"
                />
                <label
                  htmlFor="file-upload"
                  className="upload-button"
                >
                  {isProcessing ? 'Processing...' : '📁 Choose Image'}
                </label>
                <p className="upload-hint">
                  Or drag and drop an image here
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Ingredients Step
  if (currentStep === 'ingredients') {
    const groupedIngredients = groupIngredients(detectedIngredients);
    
    return (
      <div className="scanner-container">
        {/* Header Section */}
        <div className="scanner-header">
          <div className="scanner-header-content">
            <h1 className="scanner-title">Detected Ingredients</h1>
            <p className="scanner-subtitle">
              We found these ingredients in your image
            </p>
          </div>
        </div>

        {/* Main Content */}
        <div className="scanner-content">
          {/* Ingredients Section */}
          <div className="scanner-section">
            <div className="scanner-card">
              <h2 className="section-title">
                <span className="section-icon">🥕</span>
                Detected Ingredients
              </h2>
              
              <div className="ingredients-grid">
                {groupedIngredients.map((item, index) => (
                  <div key={index} className="ingredient-card">
                    <div className="ingredient-content">
                      <div className="ingredient-icon">
                        {getIngredientIcon(item.name)}
                      </div>
                      <div className="ingredient-info">
                        <div className="ingredient-header">
                          <h4 className="ingredient-name">{item.name}</h4>
                          {item.count > 1 && (
                            <span className="ingredient-count">×{item.count}</span>
                          )}
                        </div>
                        <p className="ingredient-confidence">
                          Confidence: {Math.round(item.confidence * 100)}%
                        </p>
                        <div className="confidence-bar">
                          <div 
                            className="confidence-fill" 
                            style={{ width: `${item.confidence * 100}%` }}
                          ></div>
                        </div>
                        {item.count > 1 && (
                          <p className="ingredient-detail">
                            Detected {item.count} times (max confidence: {Math.round(item.maxConfidence * 100)}%)
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="action-buttons">
            <button
              onClick={handleGenerateRecipes}
              className="primary-button"
            >
              👨‍🍳 Generate Recipes
            </button>
            <button
              onClick={() => {
                setCurrentStep('upload');
                setDetectedIngredients([]);
                setGeneratedRecipes([]);
                setAnalytics(null);
              }}
              className="secondary-button"
            >
              ← Back to Scanner
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Recipes Step
  if (currentStep === 'recipes') {
    return (
      <div className="scanner-container">
        {/* Header Section */}
        <div className="scanner-header">
          <div className="scanner-header-content">
            <h1 className="scanner-title">Generated Recipes</h1>
            <p className="scanner-subtitle">
              Personalized recipes based on your ingredients
            </p>
          </div>
        </div>

        {/* Main Content */}
        <div className="scanner-content">
          {/* Analytics Summary */}
          {analytics && (
            <div className="scanner-section">
              <div className="analytics-summary">
                <h3 className="analytics-title">
                  <span className="section-icon">📊</span>
                  Impact Summary
                </h3>
                <div className="analytics-grid">
                  <div className="analytics-metric">
                    <div className="analytics-value">{analytics.waste_reduction_potential}kg</div>
                    <div className="analytics-label">Waste Reduced</div>
                  </div>
                  <div className="analytics-metric">
                    <div className="analytics-value">{analytics.ingredients_utilized}</div>
                    <div className="analytics-label">Ingredients Used</div>
                  </div>
                  <div className="analytics-metric">
                    <div className="analytics-value">{analytics.recipes_available}</div>
                    <div className="analytics-label">Recipes Generated</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="action-buttons">
            <button
              onClick={handleViewRecipes}
              className="primary-button view"
            >
              👁️ View All Recipes
            </button>
            <button
              onClick={() => {
                setCurrentStep('upload');
                setDetectedIngredients([]);
                setGeneratedRecipes([]);
                setAnalytics(null);
              }}
              className="secondary-button"
            >
              ← Scan New Image
            </button>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default Scanner;
