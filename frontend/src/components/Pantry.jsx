import React, { useState } from 'react';

const Pantry = ({ items, onItemAdded }) => {
  const [newItem, setNewItem] = useState({ name: '', quantity: 1, unit: 'unit' });
  const [isAdding, setIsAdding] = useState(false);

  const handleAddItem = async () => {
    if (!newItem.name.trim()) return;
    
    setIsAdding(true);
    try {
      const response = await fetch('http://localhost:5000/api/pantry/items', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newItem),
      });
      
      if (response.ok) {
        setNewItem({ name: '', quantity: 1, unit: 'unit' });
        onItemAdded();
      }
    } catch (error) {
      console.error('Error adding item:', error);
    } finally {
      setIsAdding(false);
    }
  };

  const handleDeleteItem = async (itemId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/pantry/items/${itemId}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        onItemAdded();
      }
    } catch (error) {
      console.error('Error deleting item:', error);
    }
  };

  const getExpiryStatusColor = (status) => {
    switch (status) {
      case 'expired': return 'bg-red-100 text-red-800 border-red-200';
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      case 'warning': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'good': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getExpiryStatusText = (status) => {
    switch (status) {
      case 'expired': return 'Expired';
      case 'critical': return 'Expiring soon';
      case 'warning': return 'Use soon';
      case 'good': return 'Fresh';
      default: return 'Unknown';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Pantry Items</h2>
        <span className="text-sm text-gray-500">{items.length} items</span>
      </div>

      {/* Add Item Form */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Add New Item</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <input
            type="text"
            placeholder="Item name"
            value={newItem.name}
            onChange={(e) => setNewItem({ ...newItem, name: e.target.value })}
            className="col-span-2 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <input
            type="number"
            placeholder="Quantity"
            value={newItem.quantity}
            onChange={(e) => setNewItem({ ...newItem, quantity: parseFloat(e.target.value) || 1 })}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <select
            value={newItem.unit}
            onChange={(e) => setNewItem({ ...newItem, unit: e.target.value })}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="unit">unit</option>
            <option value="kg">kg</option>
            <option value="g">g</option>
            <option value="l">L</option>
            <option value="ml">ml</option>
            <option value="pcs">pcs</option>
          </select>
        </div>
        <button
          onClick={handleAddItem}
          disabled={isAdding || !newItem.name.trim()}
          className="mt-4 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isAdding ? 'Adding...' : 'Add Item'}
        </button>
      </div>

      {/* Items List */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Your Items</h3>
        </div>
        <div className="divide-y divide-gray-200">
          {items.length === 0 ? (
            <div className="p-6 text-center text-gray-500">
              <p>No items in your pantry yet.</p>
              <p className="text-sm">Add some items to get started!</p>
            </div>
          ) : (
            items.map((item) => (
              <div key={item._id} className="p-6 flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="flex-shrink-0">
                    <span className="text-2xl">
                      {item.category === 'fruits' ? '🍎' :
                       item.category === 'vegetables' ? '🥬' :
                       item.category === 'dairy' ? '🥛' :
                       item.category === 'meat' ? '🥩' :
                       item.category === 'grains' ? '🍞' :
                       item.category === 'beverages' ? '🥤' : '🥫'}
                    </span>
                  </div>
                  <div>
                    <h4 className="text-lg font-medium text-gray-900">{item.name}</h4>
                    <p className="text-sm text-gray-500">
                      {item.quantity} {item.unit}
                      {item.expiry_date && (
                        <span className="ml-2">
                          • Expires: {new Date(item.expiry_date).toLocaleDateString()}
                        </span>
                      )}
                    </p>
                    {item.confidence && (
                      <p className="text-xs text-gray-400">
                        Confidence: {Math.round(item.confidence * 100)}%
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  {item.expiry_status && (
                    <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getExpiryStatusColor(item.expiry_status)}`}>
                      {getExpiryStatusText(item.expiry_status)}
                    </span>
                  )}
                  <button
                    onClick={() => handleDeleteItem(item._id)}
                    className="text-red-600 hover:text-red-800"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default Pantry;
