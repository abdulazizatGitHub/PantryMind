import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import Scanner from './components/Scanner';
import RecipeView from './components/RecipeView';
import Navigation from './components/Navigation';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [dashboardStats, setDashboardStats] = useState({});
  const [generatedRecipes, setGeneratedRecipes] = useState([]);

  useEffect(() => {
    fetchDashboardStats();
  }, []);

  const fetchDashboardStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/analytics/waste-reduction');
      if (response.ok) {
        const data = await response.json();
        setDashboardStats(data);
      }
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
    }
  };

  const handleRecipeSelect = (recipes) => {
    setGeneratedRecipes(recipes);
    setActiveTab('recipes');
  };

  const handleAnalyticsUpdate = async (analyticsData) => {
    // Update dashboard stats with new analytics data from recipe generation
    if (analyticsData) {
      setDashboardStats(prevStats => ({
        ...prevStats,
        ...analyticsData
      }));
    }
    
    // Also refresh from server to get the latest aggregated data
    await fetchDashboardStats();
  };

  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard stats={dashboardStats} onRefresh={fetchDashboardStats} />;
      case 'scanner':
        return <Scanner onRecipeSelect={handleRecipeSelect} onAnalyticsUpdate={handleAnalyticsUpdate} />;
      case 'recipes':
        return <RecipeView recipes={generatedRecipes} onBack={() => setActiveTab('scanner')} />;
      default:
        return <Dashboard stats={dashboardStats} onRefresh={fetchDashboardStats} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="py-8">
          <h1 className="text-4xl font-bold text-center text-gray-900 mb-8">
            🍽️ PantryMind AI
          </h1>
          
          <Navigation activeTab={activeTab} onTabChange={handleTabChange} />
          
          <main className="mt-8">
            {renderContent()}
          </main>
        </div>
      </div>
    </div>
  );
}

export default App;