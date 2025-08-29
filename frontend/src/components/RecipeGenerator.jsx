import React, { useState } from 'react';
import './RecipeGenerator.css';

const RecipeGenerator = ({ pantryItems, onItemAdded }) => {
  const [recipes, setRecipes] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedItems, setSelectedItems] = useState([]);

  const generateRecipes = async () => {
    if (pantryItems.length === 0) {
      alert('No items in your pantry. Please add some items first.');
      return;
    }

    setIsGenerating(true);
    try {
      const response = await fetch('http://localhost:8000/api/recipes/suggest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pantry_items: pantryItems.map(item => item.name),
          max_recipes: 3
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setRecipes(data.top3 || []);
      } else {
        throw new Error('Failed to generate recipes');
      }
    } catch (error) {
      console.error('Error generating recipes:', error);
      alert('Error generating recipes. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const generateRecipesFromSelection = async () => {
    if (selectedItems.length === 0) {
      alert('Please select at least one item from your pantry.');
      return;
    }

    setIsGenerating(true);
    try {
      const response = await fetch('http://localhost:5000/api/recipes/suggest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pantry_items: selectedItems,
          max_recipes: 3
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setRecipes(data.recipes || []);
      } else {
        throw new Error('Failed to generate recipes');
      }
    } catch (error) {
      console.error('Error generating recipes:', error);
      alert('Error generating recipes. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleItemToggle = (itemId) => {
    setSelectedItems(prev => 
      prev.includes(itemId) 
        ? prev.filter(id => id !== itemId)
        : [...prev, itemId]
    );
  };

  const useRecipe = async (recipeId) => {
    try {
      const response = await fetch(`http://localhost:8000/api/recipes/${recipeId}/use`, {
        method: 'POST',
      });

      if (response.ok) {
        alert('Recipe marked as used! Waste reduction logged.');
        onItemAdded(); // Refresh dashboard stats
      }
    } catch (error) {
      console.error('Error using recipe:', error);
    }
  };

  return (
    <div className="recipe-generator-container">
      {/* Header Section */}
      <div className="recipe-generator-header">
        <div className="recipe-generator-header-content">
          <h1 className="recipe-generator-title">Recipe Generator</h1>
          <p className="recipe-generator-subtitle">
            Generate personalized recipes from your pantry items and reduce food waste
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="recipe-generator-content">
        {/* Pantry Items Selection */}
        {pantryItems.length > 0 && (
          <div className="recipe-generator-section">
            <div className="recipe-generator-card">
              <h2 className="section-title">
                <span className="section-icon">🥕</span>
                Select Items for Recipe Generation
              </h2>
              
              <div className="pantry-grid">
                {pantryItems.map((item) => (
                  <div
                    key={item._id}
                    className={`pantry-item ${selectedItems.includes(item._id) ? 'selected' : ''}`}
                    onClick={() => handleItemToggle(item._id)}
                  >
                    <div className="pantry-checkbox"></div>
                    <div className="pantry-item-content">
                      <div className="pantry-item-name">{item.name}</div>
                      <div className="pantry-item-details">
                        {item.quantity} {item.unit}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="action-buttons">
                <button
                  onClick={generateRecipes}
                  disabled={isGenerating || pantryItems.length === 0}
                  className="primary-button"
                >
                  {isGenerating ? 'Generating...' : '👨‍🍳 Generate Recipes'}
                </button>
                <button
                  onClick={generateRecipesFromSelection}
                  disabled={isGenerating || selectedItems.length === 0}
                  className="secondary-button"
                >
                  {isGenerating ? 'Generating...' : `Generate from ${selectedItems.length} selected items`}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Generated Recipes */}
        {recipes.length > 0 && (
          <div className="recipe-generator-section">
            <div className="recipe-generator-card">
              <div className="recipes-header">
                <h2 className="recipes-title">
                  <span className="section-icon">🍽️</span>
                  Suggested Recipes
                </h2>
                <div className="recipes-count">{recipes.length} recipes</div>
              </div>
              
              <div className="recipes-grid">
                {recipes.map((recipe, index) => (
                  <div key={recipe.id || index} className="recipe-card">
                    <div className="recipe-header">
                      <h3 className="recipe-title">{recipe.title}</h3>
                      <div className="recipe-badges">
                        <span className="recipe-badge difficulty">
                          {recipe.difficulty || 'Medium'}
                        </span>
                        <span className="recipe-badge time">
                          {recipe.cooking_time || 30} min
                        </span>
                        <span className="recipe-badge health">
                          {recipe.health_score || 75}%
                        </span>
                      </div>
                    </div>

                    <div className="recipe-content">
                      {/* Ingredients */}
                      <div className="recipe-section">
                        <h4 className="recipe-section-title">
                          <span>🥕</span>
                          Ingredients
                        </h4>
                        <div className="recipe-ingredients">
                          {recipe.ingredients.map((ingredient, idx) => (
                            <span key={idx} className="recipe-ingredient">
                              {ingredient}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Instructions */}
                      <div className="recipe-section">
                        <h4 className="recipe-section-title">
                          <span>📝</span>
                          Instructions
                        </h4>
                        <ol className="recipe-instructions">
                          {recipe.instructions.map((step, idx) => (
                            <li key={idx} className="recipe-instruction">
                              <div className="recipe-instruction-number">{idx + 1}</div>
                              <div className="recipe-instruction-text">{step}</div>
                            </li>
                          ))}
                        </ol>
                      </div>
                    </div>

                    <div className="recipe-footer">
                      <div className="recipe-stats">
                        Uses {recipe.ingredients.length} ingredients from your pantry
                      </div>
                      <button
                        onClick={() => useRecipe(recipe.id)}
                        className="use-recipe-button"
                      >
                        ✅ Use This Recipe
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {recipes.length === 0 && !isGenerating && (
          <div className="recipe-generator-section">
            <div className="empty-state">
              <div className="empty-state-icon">👨‍🍳</div>
              <h3 className="empty-state-title">
                No recipes generated yet
              </h3>
              <p className="empty-state-description">
                {pantryItems.length === 0 
                  ? 'Add some items to your pantry first, then generate recipes!'
                  : 'Click "Generate Recipes" to get cooking suggestions based on your pantry items.'
                }
              </p>
              {pantryItems.length > 0 && (
                <button
                  onClick={generateRecipes}
                  className="primary-button"
                >
                  Generate Your First Recipe
                </button>
              )}
            </div>
          </div>
        )}

        {/* Tips Section */}
        <div className="recipe-generator-section">
          <div className="tips-section">
            <div className="tips-header">
              <div className="tips-icon">💡</div>
              <h3 className="tips-title">Recipe Tips</h3>
            </div>
            <ul className="tips-list">
              <li className="tips-item">
                <div className="tips-bullet"></div>
                <span className="tips-text">Select specific items to get more targeted recipe suggestions</span>
              </li>
              <li className="tips-item">
                <div className="tips-bullet"></div>
                <span className="tips-text">Use recipes to reduce food waste and save money</span>
              </li>
              <li className="tips-item">
                <div className="tips-bullet"></div>
                <span className="tips-text">Mark recipes as used to track your waste reduction progress</span>
              </li>
              <li className="tips-item">
                <div className="tips-bullet"></div>
                <span className="tips-text">Check the dashboard to see your impact over time</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecipeGenerator;
