import React, { useState } from 'react';

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
      const response = await fetch('http://localhost:5000/api/recipes/suggest-from-pantry', {
        method: 'GET',
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
      const response = await fetch(`http://localhost:5000/api/recipes/recipes/${recipeId}/use`, {
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

  const getDifficultyColor = (difficulty) => {
    switch (difficulty?.toLowerCase()) {
      case 'easy': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'hard': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Recipe Generator</h2>
        <button
          onClick={generateRecipes}
          disabled={isGenerating || pantryItems.length === 0}
          className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isGenerating ? 'Generating...' : '👨‍🍳 Generate Recipes'}
        </button>
      </div>

      {/* Pantry Items Selection */}
      {pantryItems.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Select Items for Recipe Generation
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            {pantryItems.map((item) => (
              <label
                key={item._id}
                className={`flex items-center p-3 border rounded-lg cursor-pointer transition-colors ${
                  selectedItems.includes(item._id)
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedItems.includes(item._id)}
                  onChange={() => handleItemToggle(item._id)}
                  className="mr-2"
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {item.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {item.quantity} {item.unit}
                  </p>
                </div>
              </label>
            ))}
          </div>
          <button
            onClick={generateRecipesFromSelection}
            disabled={isGenerating || selectedItems.length === 0}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? 'Generating...' : `Generate from ${selectedItems.length} selected items`}
          </button>
        </div>
      )}

      {/* Generated Recipes */}
      {recipes.length > 0 && (
        <div className="space-y-6">
          <h3 className="text-xl font-bold text-gray-900">
            Suggested Recipes ({recipes.length})
          </h3>
          {recipes.map((recipe, index) => (
            <div key={recipe.id || index} className="bg-white rounded-lg shadow overflow-hidden">
              <div className="p-6">
                <div className="flex justify-between items-start mb-4">
                  <h4 className="text-xl font-bold text-gray-900">{recipe.title}</h4>
                  <div className="flex space-x-2">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getDifficultyColor(recipe.difficulty)}`}>
                      {recipe.difficulty || 'Medium'}
                    </span>
                    <span className="px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-800">
                      {recipe.cooking_time || 30} min
                    </span>
                  </div>
                </div>

                {/* Ingredients */}
                <div className="mb-4">
                  <h5 className="text-sm font-medium text-gray-700 mb-2">Ingredients:</h5>
                  <div className="flex flex-wrap gap-2">
                    {recipe.ingredients.map((ingredient, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-1 bg-gray-100 text-gray-700 text-sm rounded"
                      >
                        {ingredient}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Instructions */}
                <div className="mb-4">
                  <h5 className="text-sm font-medium text-gray-700 mb-2">Instructions:</h5>
                  <ol className="list-decimal list-inside space-y-1 text-sm text-gray-600">
                    {recipe.instructions.map((step, idx) => (
                      <li key={idx}>{step}</li>
                    ))}
                  </ol>
                </div>

                {/* Actions */}
                <div className="flex justify-between items-center pt-4 border-t border-gray-200">
                  <div className="text-sm text-gray-500">
                    Uses {recipe.ingredients.length} ingredients from your pantry
                  </div>
                  <button
                    onClick={() => useRecipe(recipe.id)}
                    className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 text-sm"
                  >
                    ✅ Use This Recipe
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {recipes.length === 0 && !isGenerating && (
        <div className="bg-white p-12 rounded-lg shadow text-center">
          <div className="text-6xl mb-4">👨‍🍳</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No recipes generated yet
          </h3>
          <p className="text-gray-500 mb-6">
            {pantryItems.length === 0 
              ? 'Add some items to your pantry first, then generate recipes!'
              : 'Click "Generate Recipes" to get cooking suggestions based on your pantry items.'
            }
          </p>
          {pantryItems.length > 0 && (
            <button
              onClick={generateRecipes}
              className="bg-green-600 text-white px-6 py-3 rounded-md hover:bg-green-700"
            >
              Generate Your First Recipe
            </button>
          )}
        </div>
      )}

      {/* Tips */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-medium text-blue-900 mb-3">💡 Recipe Tips</h3>
        <ul className="space-y-2 text-blue-800">
          <li>• Select specific items to get more targeted recipe suggestions</li>
          <li>• Use recipes to reduce food waste and save money</li>
          <li>• Mark recipes as used to track your waste reduction progress</li>
          <li>• Check the dashboard to see your impact over time</li>
        </ul>
      </div>
    </div>
  );
};

export default RecipeGenerator;
