import React from 'react';

const Dashboard = ({ stats, onRefresh }) => {
  const {
    pantry_items = 0,
    expiring_soon = 0,
    waste_reduced_kg = 0,
    money_saved_usd = 0,
    recent_interactions = []
  } = stats;

  const StatCard = ({ title, value, subtitle, icon, color }) => (
    <div className={`bg-white overflow-hidden shadow rounded-lg ${color}`}>
      <div className="p-5">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <span className="text-2xl">{icon}</span>
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">{title}</dt>
              <dd className="text-lg font-medium text-gray-900">{value}</dd>
              {subtitle && <dd className="text-sm text-gray-500">{subtitle}</dd>}
            </dl>
          </div>
        </div>
      </div>
    </div>
  );

  const InteractionItem = ({ interaction }) => {
    const getActionIcon = (action) => {
      switch (action) {
        case 'scan_item': return '📷';
        case 'add_item': return '➕';
        case 'delete_item': return '🗑️';
        case 'generate_recipes': return '👨‍🍳';
        case 'use_recipe': return '✅';
        case 'waste_reduced': return '🌱';
        default: return '📝';
      }
    };

    const getActionText = (action) => {
      switch (action) {
        case 'scan_item': return 'Scanned food item';
        case 'add_item': return 'Added item to pantry';
        case 'delete_item': return 'Removed item from pantry';
        case 'generate_recipes': return 'Generated recipes';
        case 'use_recipe': return 'Used recipe';
        case 'waste_reduced': return 'Reduced food waste';
        default: return action;
      }
    };

    return (
      <div className="flex items-center space-x-3 p-3 bg-white rounded-lg shadow-sm">
        <span className="text-xl">{getActionIcon(interaction.action)}</span>
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-900">{getActionText(interaction.action)}</p>
          <p className="text-xs text-gray-500">
            {new Date(interaction.created_at).toLocaleString()}
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>
        <button
          onClick={onRefresh}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
        >
          🔄 Refresh
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Pantry Items"
          value={pantry_items}
          subtitle="Total items"
          icon="🥫"
          color="border-l-4 border-blue-400"
        />
        <StatCard
          title="Expiring Soon"
          value={expiring_soon}
          subtitle="Within 7 days"
          icon="⚠️"
          color="border-l-4 border-yellow-400"
        />
        <StatCard
          title="Waste Reduced"
          value={`${waste_reduced_kg} kg`}
          subtitle="Total saved"
          icon="🌱"
          color="border-l-4 border-green-400"
        />
        <StatCard
          title="Money Saved"
          value={`$${money_saved_usd}`}
          subtitle="Estimated savings"
          icon="💰"
          color="border-l-4 border-purple-400"
        />
      </div>

      {/* Recent Activity */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Recent Activity</h3>
        </div>
        <div className="p-6">
          {recent_interactions.length > 0 ? (
            <div className="space-y-3">
              {recent_interactions.map((interaction, index) => (
                <InteractionItem key={index} interaction={interaction} />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">No recent activity</p>
          )}
        </div>
      </div>

      {/* Tips */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-medium text-blue-900 mb-3">💡 Tips to Reduce Food Waste</h3>
        <ul className="space-y-2 text-blue-800">
          <li>• Scan items when you buy them to track expiry dates</li>
          <li>• Use the recipe generator to find ways to use expiring items</li>
          <li>• Check your pantry regularly for items that need to be used soon</li>
          <li>• Plan meals around items that are expiring first</li>
        </ul>
      </div>
    </div>
  );
};

export default Dashboard;
