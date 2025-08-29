import React, { useState, useMemo, useEffect } from 'react';

const Pantry = ({ items = [], onItemAdded }) => {
  const [pantryItems, setPantryItems] = useState([]);
  const [newItem, setNewItem] = useState({ name: '', quantity: 1, unit: 'unit' });
  const [isAdding, setIsAdding] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [sortBy, setSortBy] = useState('name');

  useEffect(() => {
    fetchPantryItems();
  }, []);

  const fetchPantryItems = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/pantry/list');
      if (response.ok) {
        const data = await response.json();
        setPantryItems(data);
      }
    } catch (error) {
      console.error('Error fetching pantry items:', error);
    }
  };

  // Use real data or fallback to mock data
  const mockItems = pantryItems.length === 0 ? [
    {
      _id: '1',
      name: 'Fresh Apples',
      quantity: 5,
      unit: 'pcs',
      category: 'fruits',
      expiry_date: '2024-12-15',
      expiry_status: 'good',
      confidence: 0.95
    },
    {
      _id: '2',
      name: 'Whole Milk',
      quantity: 1,
      unit: 'L',
      category: 'dairy',
      expiry_date: '2024-12-12',
      expiry_status: 'warning',
      confidence: 0.88
    },
    {
      _id: '3',
      name: 'Bread Loaf',
      quantity: 1,
      unit: 'unit',
      category: 'grains',
      expiry_date: '2024-12-10',
      expiry_status: 'critical',
      confidence: 0.92
    },
    {
      _id: '4',
      name: 'Chicken Breast',
      quantity: 0.8,
      unit: 'kg',
      category: 'meat',
      expiry_date: '2024-12-08',
      expiry_status: 'expired',
      confidence: 0.85
    },
    {
      _id: '5',
      name: 'Orange Juice',
      quantity: 1,
      unit: 'L',
      category: 'beverages',
      expiry_date: '2024-12-20',
      expiry_status: 'good',
      confidence: 0.91
    }
  ] : items;

  const handleAddItem = async () => {
    if (!newItem.name.trim()) return;
    
    setIsAdding(true);
    try {
      const response = await fetch('http://localhost:8000/api/pantry/upsert', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newItem),
      });

      if (response.ok) {
        setNewItem({ name: '', quantity: 1, unit: 'unit' });
        await fetchPantryItems(); // Refresh the list
        if (onItemAdded) onItemAdded();
      } else {
        throw new Error('Failed to add item');
      }
    } catch (error) {
      console.error('Error adding item:', error);
      alert('Error adding item. Please try again.');
    } finally {
      setIsAdding(false);
    }
  };

  const handleDeleteItem = async (itemId) => {
    if (!window.confirm('Are you sure you want to delete this item?')) return;
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      if (onItemAdded) onItemAdded();
    } catch (error) {
      console.error('Error deleting item:', error);
    }
  };

  const getExpiryStatusColor = (status) => {
    const colors = {
      expired: 'bg-red-100 text-red-800 border-red-200',
      critical: 'bg-orange-100 text-orange-800 border-orange-200',
      warning: 'bg-amber-100 text-amber-800 border-amber-200',
      good: 'bg-emerald-100 text-emerald-800 border-emerald-200'
    };
    return colors[status] || 'bg-slate-100 text-slate-800 border-slate-200';
  };

  const getExpiryStatusText = (status) => {
    const texts = {
      expired: 'Expired',
      critical: 'Critical',
      warning: 'Use Soon',
      good: 'Fresh'
    };
    return texts[status] || 'Unknown';
  };

  const getCategoryIcon = (category) => {
    const icons = {
      fruits: '🍎',
      vegetables: '🥬',
      dairy: '🥛',
      meat: '🥩',
      grains: '🍞',
      beverages: '🥤',
      snacks: '🍿',
      condiments: '🧂'
    };
    return icons[category] || '🥫';
  };

  const getCategoryColor = (category) => {
    const colors = {
      fruits: 'bg-red-50 text-red-700 border border-red-200',
      vegetables: 'bg-green-50 text-green-700 border border-green-200',
      dairy: 'bg-blue-50 text-blue-700 border border-blue-200',
      meat: 'bg-rose-50 text-rose-700 border border-rose-200',
      grains: 'bg-amber-50 text-amber-700 border border-amber-200',
      beverages: 'bg-purple-50 text-purple-700 border border-purple-200',
      snacks: 'bg-orange-50 text-orange-700 border border-orange-200',
      condiments: 'bg-slate-50 text-slate-700 border border-slate-200'
    };
    return colors[category] || 'bg-slate-50 text-slate-700 border border-slate-200';
  };

  const filteredAndSortedItems = useMemo(() => {
    let filtered = mockItems;

    if (searchTerm) {
      filtered = filtered.filter(item => 
        item.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (filterStatus !== 'all') {
      filtered = filtered.filter(item => item.expiry_status === filterStatus);
    }

    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'expiry':
          if (!a.expiry_date && !b.expiry_date) return 0;
          if (!a.expiry_date) return 1;
          if (!b.expiry_date) return -1;
          return new Date(a.expiry_date) - new Date(b.expiry_date);
        case 'quantity':
          return b.quantity - a.quantity;
        case 'status':
          const statusOrder = { 'expired': 0, 'critical': 1, 'warning': 2, 'good': 3 };
          return (statusOrder[a.expiry_status] || 4) - (statusOrder[b.expiry_status] || 4);
        default:
          return 0;
      }
    });

    return filtered;
  }, [mockItems, searchTerm, filterStatus, sortBy]);

  const stats = useMemo(() => {
    const total = mockItems.length;
    const expiringSoon = mockItems.filter(item => 
      item.expiry_status === 'critical' || item.expiry_status === 'warning'
    ).length;
    const expired = mockItems.filter(item => item.expiry_status === 'expired').length;
    const fresh = mockItems.filter(item => item.expiry_status === 'good').length;

    return { total, expiringSoon, expired, fresh };
  }, [mockItems]);

  const StatCard = ({ title, value, icon, color, subtitle }) => (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 hover:shadow-lg hover:border-slate-300 transition-all duration-300">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-slate-600 mb-1">{title}</p>
          <p className={`text-2xl font-bold ${color}`}>{value}</p>
          {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
        </div>
        <div className="w-12 h-12 bg-slate-50 rounded-xl flex items-center justify-center border border-slate-200">
          <span className="text-xl">{icon}</span>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {/* Header */}
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h1 className="text-3xl font-bold text-slate-900 mb-2">Pantry Management</h1>
              <p className="text-slate-600">
                Manage your food inventory and track expiration dates efficiently.
              </p>
            </div>
            <div className="mt-4 lg:mt-0 flex items-center space-x-6 text-sm">
              <div className="text-center">
                <div className="text-xl font-bold text-slate-900">{stats.total}</div>
                <div className="text-slate-500">Total Items</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-amber-600">{stats.expiringSoon}</div>
                <div className="text-slate-500">Need Attention</div>
              </div>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatCard
              title="Total Items"
              value={stats.total}
              icon="📦"
              color="text-slate-900"
            />
            <StatCard
              title="Fresh Items"
              value={stats.fresh}
              icon="✅"
              color="text-emerald-600"
            />
            <StatCard
              title="Need Attention"
              value={stats.expiringSoon}
              icon="⚠️"
              color="text-amber-600"
            />
            <StatCard
              title="Expired"
              value={stats.expired}
              icon="❌"
              color="text-red-600"
            />
          </div>

          {/* Add Item Form */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm">
            <div className="border-b border-slate-200 px-6 py-4">
              <h3 className="text-lg font-semibold text-slate-900">Add New Item</h3>
              <p className="text-sm text-slate-600 mt-1">Add items to your pantry inventory</p>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-slate-700 mb-2">Item Name</label>
                  <input
                    type="text"
                    placeholder="Enter item name..."
                    value={newItem.name}
                    onChange={(e) => setNewItem({ ...newItem, name: e.target.value })}
                    className="w-full px-3 py-2.5 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500 focus:border-transparent transition-colors duration-200"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Quantity</label>
                  <input
                    type="number"
                    placeholder="1"
                    value={newItem.quantity}
                    onChange={(e) => setNewItem({ ...newItem, quantity: parseFloat(e.target.value) || 1 })}
                    className="w-full px-3 py-2.5 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500 focus:border-transparent transition-colors duration-200"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Unit</label>
                  <select
                    value={newItem.unit}
                    onChange={(e) => setNewItem({ ...newItem, unit: e.target.value })}
                    className="w-full px-3 py-2.5 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500 focus:border-transparent transition-colors duration-200"
                  >
                    <option value="unit">unit</option>
                    <option value="kg">kg</option>
                    <option value="g">g</option>
                    <option value="l">L</option>
                    <option value="ml">ml</option>
                    <option value="pcs">pcs</option>
                  </select>
                </div>
              </div>
              <div className="mt-6">
                <button
                  onClick={handleAddItem}
                  disabled={isAdding || !newItem.name.trim()}
                  className="inline-flex items-center space-x-2 px-6 py-2.5 bg-slate-900 text-white rounded-xl hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 font-medium"
                >
                  {isAdding ? (
                    <>
                      <span className="animate-spin">↻</span>
                      <span>Adding...</span>
                    </>
                  ) : (
                    <>
                      <span>+</span>
                      <span>Add Item</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Filters and Search */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Search Items</label>
                <div className="relative">
                  <input
                    type="text"
                    placeholder="Search by name..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2.5 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500 focus:border-transparent transition-colors duration-200"
                  />
                  <div className="absolute left-3 top-3 text-slate-400">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                    </svg>
                  </div>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Filter by Status</label>
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="w-full px-3 py-2.5 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500 focus:border-transparent transition-colors duration-200"
                >
                  <option value="all">All Items</option>
                  <option value="good">Fresh</option>
                  <option value="warning">Use Soon</option>
                  <option value="critical">Critical</option>
                  <option value="expired">Expired</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Sort By</label>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="w-full px-3 py-2.5 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-500 focus:border-transparent transition-colors duration-200"
                >
                  <option value="name">Name</option>
                  <option value="expiry">Expiry Date</option>
                  <option value="quantity">Quantity</option>
                  <option value="status">Status</option>
                </select>
              </div>
            </div>
          </div>

          {/* Items List */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm">
            <div className="border-b border-slate-200 px-6 py-4">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">Inventory Items</h3>
                  <p className="text-sm text-slate-600 mt-1">
                    {filteredAndSortedItems.length} of {mockItems.length} items
                  </p>
                </div>
                <div className="flex items-center space-x-2 text-xs">
                  <span className="text-slate-500">Showing:</span>
                  <span className="bg-slate-100 text-slate-700 px-2.5 py-1 rounded-lg font-medium">
                    {filteredAndSortedItems.length} items
                  </span>
                </div>
              </div>
            </div>
            
            <div className="p-6">
              {filteredAndSortedItems.length === 0 ? (
                <div className="text-center py-16">
                  <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <span className="text-2xl">📦</span>
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900 mb-2">
                    {mockItems.length === 0 ? 'No items in your pantry' : 'No items match your criteria'}
                  </h3>
                  <p className="text-slate-600 mb-6">
                    {mockItems.length === 0 
                      ? 'Add some items to get started with inventory management.'
                      : 'Try adjusting your search or filter criteria.'
                    }
                  </p>
                  {mockItems.length === 0 && (
                    <button
                      onClick={() => document.querySelector('input[type="text"]')?.focus()}
                      className="inline-flex items-center space-x-2 px-6 py-2.5 bg-slate-900 text-white rounded-xl hover:bg-slate-800 transition-colors duration-200 font-medium"
                    >
                      <span>+</span>
                      <span>Add Your First Item</span>
                    </button>
                  )}
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {filteredAndSortedItems.map((item) => (
                    <div key={item._id} className="group bg-slate-50 border border-slate-200 rounded-2xl p-6 hover:border-slate-300 hover:shadow-lg transition-all duration-300">
                      <div className="flex items-start justify-between mb-4">
                        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${getCategoryColor(item.category)}`}>
                          <span className="text-xl">{getCategoryIcon(item.category)}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          {item.expiry_status && (
                            <span className={`px-2.5 py-1 text-xs font-semibold rounded-lg border ${getExpiryStatusColor(item.expiry_status)}`}>
                              {getExpiryStatusText(item.expiry_status)}
                            </span>
                          )}
                          <button
                            onClick={() => handleDeleteItem(item._id)}
                            className="text-slate-400 hover:text-red-500 transition-colors duration-200 p-1.5 hover:bg-red-50 rounded-lg"
                            title="Delete item"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                            </svg>
                          </button>
                        </div>
                      </div>
                      
                      <h4 className="text-lg font-semibold text-slate-900 mb-3 truncate">{item.name}</h4>
                      
                      <div className="space-y-2 mb-4">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-slate-600">Quantity:</span>
                          <span className="font-semibold text-slate-900">
                            {item.quantity} {item.unit}
                          </span>
                        </div>
                        
                        {item.expiry_date && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-slate-600">Expires:</span>
                            <span className={`font-semibold ${
                              item.expiry_status === 'expired' ? 'text-red-600' :
                              item.expiry_status === 'critical' ? 'text-orange-600' :
                              item.expiry_status === 'warning' ? 'text-amber-600' :
                              'text-emerald-600'
                            }`}>
                              {new Date(item.expiry_date).toLocaleDateString()}
                            </span>
                          </div>
                        )}
                        
                        {item.confidence && (
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-slate-600">Accuracy:</span>
                            <span className="font-semibold text-slate-900">
                              {Math.round(item.confidence * 100)}%
                            </span>
                          </div>
                        )}
                      </div>
                      
                      {item.expiry_date && (
                        <div className="w-full bg-slate-200 rounded-full h-1.5">
                          <div 
                            className={`h-1.5 rounded-full transition-all duration-1000 ${
                              item.expiry_status === 'expired' ? 'bg-red-500' :
                              item.expiry_status === 'critical' ? 'bg-orange-500' :
                              item.expiry_status === 'warning' ? 'bg-amber-500' :
                              'bg-emerald-500'
                            }`}
                            style={{ 
                              width: `${Math.max(10, Math.min(100, 
                                ((new Date(item.expiry_date) - new Date()) / (7 * 24 * 60 * 60 * 1000)) * 100
                              ))}%` 
                            }}
                          ></div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Pantry;