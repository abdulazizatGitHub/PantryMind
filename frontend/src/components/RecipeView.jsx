import React, { useState } from 'react';
import './RecipeView.css';

const RecipeView = ({ recipes, onBack }) => {
  const [selectedRecipe, setSelectedRecipe] = useState(null);

  const handleRecipeClick = (recipe) => {
    setSelectedRecipe(recipe);
  };

  const handleBackToRecipes = () => {
    setSelectedRecipe(null);
  };

  if (!recipes || recipes.length === 0) {
    return (
      <div className="recipe-view-container">
        <div className="recipe-view-content">
          <div className="error-state">
            <h2 className="error-title">No Recipes Found</h2>
            <p className="error-description">
              No recipes were generated. Please try scanning ingredients again.
            </p>
            <button className="action-button" onClick={onBack}>
              ← Back to Scanner
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (selectedRecipe) {
    return (
      <div className="recipe-view-container">
        <div className="recipe-detail-container">
          {/* Recipe Detail Header */}
          <div className="recipe-detail-header">
            <h1 className="recipe-detail-title">{selectedRecipe.title}</h1>
            <p className="recipe-detail-subtitle">{selectedRecipe.description}</p>
            
            {/* Recipe Stats */}
            <div className="recipe-detail-stats">
              <div className="recipe-detail-stat">
                <div className="recipe-detail-stat-value">{selectedRecipe.cooking_time || '30'}</div>
                <div className="recipe-detail-stat-label">Minutes</div>
              </div>
              <div className="recipe-detail-stat">
                <div className="recipe-detail-stat-value">{selectedRecipe.servings || '4'}</div>
                <div className="recipe-detail-stat-label">Servings</div>
              </div>
              <div className="recipe-detail-stat">
                <div className="recipe-detail-stat-value">{selectedRecipe.difficulty || 'Medium'}</div>
                <div className="recipe-detail-stat-label">Difficulty</div>
              </div>
              <div className="recipe-detail-stat">
                <div className="recipe-detail-stat-value">{selectedRecipe.health_score || '85'}%</div>
                <div className="recipe-detail-stat-label">Health Score</div>
              </div>
            </div>

            {/* Recipe Tags */}
            <div className="recipe-detail-tags">
              {selectedRecipe.difficulty && (
                <span className="recipe-detail-tag difficulty">{selectedRecipe.difficulty}</span>
              )}
              {selectedRecipe.cuisine && (
                <span className="recipe-detail-tag cuisine">{selectedRecipe.cuisine}</span>
              )}
              {selectedRecipe.health_score && (
                <span className="recipe-detail-tag health">Healthy</span>
              )}
              {selectedRecipe.tags && selectedRecipe.tags.map((tag, index) => (
                <span key={index} className="recipe-detail-tag general">{tag}</span>
              ))}
            </div>
          </div>

          {/* Main Content */}
          <div className="recipe-main-content">
            <div className="recipe-content-grid">
              {/* Left Column - Instructions */}
              <div className="recipe-instructions-section">
                <h2 className="instructions-title">📝 Cooking Instructions</h2>
                <div className="instructions-list">
                  {selectedRecipe.instructions && selectedRecipe.instructions.map((instruction, index) => (
                    <div key={index} className="instruction-step">
                      <div className="instruction-number">{index + 1}</div>
                      <div className="instruction-text">{instruction}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Right Column - Ingredients and Info */}
              <div className="recipe-sidebar">
                {/* Ingredients Section */}
                <div className="ingredients-section">
                  <h3 className="ingredients-title">🥕 Ingredients</h3>
                  
                  {selectedRecipe.core_ingredients && selectedRecipe.core_ingredients.length > 0 && (
                    <div className="ingredients-subsection">
                      <h4 className="ingredients-subsection-title">Core Ingredients</h4>
                      <div className="ingredients-grid">
                        {selectedRecipe.core_ingredients.map((ingredient, index) => (
                          <div key={index} className="ingredient-item available">
                            <div className="ingredient-icon">🥕</div>
                            <div className="ingredient-name">{ingredient}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {selectedRecipe.additional_ingredients && selectedRecipe.additional_ingredients.length > 0 && (
                    <div className="ingredients-subsection">
                      <h4 className="ingredients-subsection-title">Additional Ingredients</h4>
                      <div className="ingredients-grid">
                        {selectedRecipe.additional_ingredients.map((ingredient, index) => (
                          <div key={index} className="ingredient-item missing">
                            <div className="ingredient-icon">🧂</div>
                            <div className="ingredient-name">{ingredient}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Nutritional Information */}
                <div className="nutritional-section">
                  <h3 className="nutritional-title">📊 Nutritional Info</h3>
                  <div className="nutritional-grid">
                    <div className="nutritional-item">
                      <div className="nutritional-value">{selectedRecipe.calories || '350'}</div>
                      <div className="nutritional-label">Calories</div>
                    </div>
                    <div className="nutritional-item">
                      <div className="nutritional-value">{selectedRecipe.protein || '12'}</div>
                      <div className="nutritional-label">Protein (g)</div>
                    </div>
                    <div className="nutritional-item">
                      <div className="nutritional-value">{selectedRecipe.carbs || '45'}</div>
                      <div className="nutritional-label">Carbs (g)</div>
                    </div>
                    <div className="nutritional-item">
                      <div className="nutritional-value">{selectedRecipe.fat || '8'}</div>
                      <div className="nutritional-label">Fat (g)</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Environmental Impact Section */}
            <div className="environmental-section">
              <h3 className="environmental-title">🌱 Environmental Impact</h3>
              <div className="environmental-grid">
                <div className="environmental-item">
                  <div className="environmental-value">{selectedRecipe.waste_reduction || '75'}%</div>
                  <div className="environmental-label">Waste Reduction</div>
                  <div className="environmental-description">Reduces food waste by using available ingredients</div>
                </div>
                <div className="environmental-item">
                  <div className="environmental-value">{selectedRecipe.carbon_footprint || '2.3'}kg</div>
                  <div className="environmental-label">Carbon Footprint</div>
                  <div className="environmental-description">Lower than average recipe carbon impact</div>
                </div>
                <div className="environmental-item">
                  <div className="environmental-value">{selectedRecipe.sustainability_score || '85'}%</div>
                  <div className="environmental-label">Sustainability Score</div>
                  <div className="environmental-description">High sustainability rating</div>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="action-buttons">
              <button className="action-button" onClick={handleBackToRecipes}>
                ← Back to Recipes
              </button>
              <button className="action-button save">
                💾 Save Recipe
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="recipe-view-container">
      <div className="recipe-view-content">
        {/* Navigation Header */}
        <div className="nav-header">
          <button className="back-button" onClick={onBack}>
            ← Back to Scanner
          </button>
          <h1 className="page-title">Generated Recipes</h1>
        </div>

        {/* Recipes Grid */}
        <div className="recipes-grid">
          {recipes.map((recipe, index) => (
            <div key={index} className="recipe-card" onClick={() => handleRecipeClick(recipe)}>
              <div className="recipe-card-header">
                <h3 className="recipe-card-title">{recipe.title}</h3>
                <div className="recipe-card-badges">
                  {recipe.difficulty && (
                    <span className="recipe-card-badge difficulty">{recipe.difficulty}</span>
                  )}
                  {recipe.cooking_time && (
                    <span className="recipe-card-badge time">{recipe.cooking_time} min</span>
                  )}
                  {recipe.health_score && (
                    <span className="recipe-card-badge health">Healthy</span>
                  )}
                </div>
                <p className="recipe-card-description">{recipe.description}</p>
              </div>
              
              <div className="recipe-card-content">
                <div className="recipe-card-section">
                  <h4 className="recipe-card-section-title">🥕 Core Ingredients</h4>
                  <div className="recipe-card-ingredients">
                    {recipe.core_ingredients && recipe.core_ingredients.slice(0, 5).map((ingredient, idx) => (
                      <span key={idx} className="recipe-card-ingredient available">
                        {ingredient}
                      </span>
                    ))}
                    {recipe.core_ingredients && recipe.core_ingredients.length > 5 && (
                      <span className="recipe-card-ingredient">+{recipe.core_ingredients.length - 5} more</span>
                    )}
                  </div>
                </div>

                <div className="recipe-card-stats">
                  <span>⏱️ {recipe.cooking_time || '30'} min</span>
                  <span>👥 {recipe.servings || '4'} servings</span>
                  <span>⭐ {recipe.health_score || '85'}% health</span>
                </div>

                <button className="view-recipe-button">
                  View Full Recipe →
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default RecipeView;