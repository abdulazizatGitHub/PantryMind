import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import Pantry from './components/Pantry';
import RecipeGenerator from './components/RecipeGenerator';
import Scanner from './components/Scanner';
import Navigation from './components/Navigation';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [pantryItems, setPantryItems] = useState([]);
  const [dashboardStats, setDashboardStats] = useState({});

  useEffect(() => {
    fetchPantryItems();
    fetchDashboardStats();
  }, []);

  const fetchPantryItems = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/pantry/items');
      const data = await response.json();
      setPantryItems(data.items || []);
    } catch (error) {
      console.error('Error fetching pantry items:', error);
    }
  };

  const fetchDashboardStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/dashboard');
      const data = await response.json();
      setDashboardStats(data);
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
    }
  };

  const handleItemAdded = () => {
    fetchPantryItems();
    fetchDashboardStats();
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard stats={dashboardStats} onRefresh={fetchDashboardStats} />;
      case 'pantry':
        return <Pantry items={pantryItems} onItemAdded={handleItemAdded} />;
      case 'scanner':
        return <Scanner onItemAdded={handleItemAdded} />;
      case 'recipes':
        return <RecipeGenerator pantryItems={pantryItems} onItemAdded={handleItemAdded} />;
      default:
        return <Dashboard stats={dashboardStats} onRefresh={fetchDashboardStats} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="py-8">
          <h1 className="text-4xl font-bold text-center text-gray-900 mb-8">
            🍽️ AI Food Waste Reducer
          </h1>
          
          <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
          
          <main className="mt-8">
            {renderContent()}
          </main>
        </div>
      </div>
    </div>
  );
}

export default App;