import React from 'react';

const Navigation = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'dashboard', name: 'Dashboard', icon: '📊' },
    { id: 'pantry', name: 'Pantry', icon: '🥫' },
    { id: 'scanner', name: 'Scanner', icon: '📷' },
    { id: 'recipes', name: 'Recipes', icon: '👨‍🍳' }
  ];

  return (
    <nav className="bg-white shadow-lg rounded-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`inline-flex items-center px-4 py-2 border-b-2 font-medium text-sm leading-5 transition-colors duration-200 ease-in-out ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.name}
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
