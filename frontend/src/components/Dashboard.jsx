import React, { useState, useEffect } from 'react';
import './Dashboard.css';

const Dashboard = ({ stats, onRefresh }) => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [analytics, setAnalytics] = useState(null);
  const [recipeStats, setRecipeStats] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  // Update analytics when stats prop changes (from parent component)
  useEffect(() => {
    if (stats && Object.keys(stats).length > 0) {
      setAnalytics(stats);
      setLastUpdated(new Date());
    }
  }, [stats]);

  const fetchAnalytics = async () => {
    try {
      // Fetch waste reduction analytics
      const wasteResponse = await fetch('http://localhost:8000/api/analytics/waste-reduction');
      if (wasteResponse.ok) {
        const wasteData = await wasteResponse.json();
        setAnalytics(wasteData);
        setLastUpdated(new Date());
      }

      // Fetch recipe statistics
      const recipeResponse = await fetch('http://localhost:8000/api/analytics/recipe-stats');
      if (recipeResponse.ok) {
        const recipeData = await recipeResponse.json();
        setRecipeStats(recipeData);
      }
    } catch (error) {
      console.error('Error fetching analytics:', error);
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchAnalytics();
    if (onRefresh) await onRefresh();
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  const formatLastUpdated = (date) => {
    if (!date) return 'Never';
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minutes ago`;
    if (diffHours < 24) return `${diffHours} hours ago`;
    return date.toLocaleDateString();
  };

  const StatCard = ({ title, value, subtitle, icon, color, trend, trendValue, isUpdated = false }) => (
    <div className={`stat-card ${isUpdated ? 'stat-card-updated' : ''}`}>
      <div className="stat-card-content">
        <div className="stat-info">
          <div className={`stat-icon ${color}`}>
            <span>{icon}</span>
          </div>
          <p className="stat-title">{title}</p>
          <p className="stat-value">{value}</p>
          <p className="stat-subtitle">{subtitle}</p>
          {trend && (
            <div className={`trend-badge ${trend === 'up' ? 'trend-up' : 'trend-down'}`}>
              <span className="mr-1">{trend === 'up' ? '↗' : '↘'}</span>
              {trendValue}
            </div>
          )}
          {isUpdated && (
            <div className="update-indicator">
              <span className="update-dot"></span>
              <span className="update-text">Updated</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="dashboard-container">
      {/* Header Section */}
      <div className="dashboard-header">
        <div className="dashboard-header-content">
          <h1 className="dashboard-title">Analytics Dashboard</h1>
          <p className="dashboard-subtitle">
            Track your food waste reduction progress and environmental impact
          </p>
          <div className="dashboard-controls">
            <div className="last-updated">
              <span className="last-updated-label">Last updated:</span>
              <span className="last-updated-time">{formatLastUpdated(lastUpdated)}</span>
            </div>
            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="refresh-button"
            >
              <span className={`refresh-icon ${isRefreshing ? 'animate-spin' : ''}`}>↻</span>
              <span>{isRefreshing ? 'Refreshing...' : 'Refresh Data'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="dashboard-content">
        {/* Key Metrics Section */}
        <div className="dashboard-section">
          <div className="section-header">
            <h2 className="section-title">
              <span className="section-icon">📊</span>
              Key Metrics
            </h2>
            {lastUpdated && (
              <p className="section-subtitle">
                Real-time data from your recipe generation activities
              </p>
            )}
          </div>
          
          <div className="stats-grid">
            <StatCard
              title="Waste Reduced"
              value={`${analytics?.waste_reduction?.total_kg || 0} kg`}
              subtitle="Total saved from landfill"
              icon="🌱"
              color="bg-emerald-50 text-emerald-700 border border-emerald-200"
              trend="up"
              trendValue={`+${analytics?.waste_reduction?.recent_kg || 0}kg`}
              isUpdated={lastUpdated && (new Date() - lastUpdated) < 300000} // 5 minutes
            />
            <StatCard
              title="CO₂ Saved"
              value={`${analytics?.waste_reduction?.co2_saved_kg || 0} kg`}
              subtitle="Environmental impact"
              icon="🌍"
              color="bg-teal-50 text-teal-700 border border-teal-200"
              trend="up"
              trendValue={`+${analytics?.waste_reduction?.recent_co2_kg || 0}kg`}
              isUpdated={lastUpdated && (new Date() - lastUpdated) < 300000}
            />
            <StatCard
              title="Money Saved"
              value={`$${analytics?.waste_reduction?.money_saved_usd || 0}`}
              subtitle="Estimated savings"
              icon="💰"
              color="bg-violet-50 text-violet-700 border border-violet-200"
              trend="up"
              trendValue={`+$${analytics?.waste_reduction?.recent_money_usd || 0}`}
              isUpdated={lastUpdated && (new Date() - lastUpdated) < 300000}
            />
            <StatCard
              title="Recipes Generated"
              value={analytics?.usage_stats?.recipes_generated || 0}
              subtitle="Total recipes created"
              icon="👨‍🍳"
              color="bg-blue-50 text-blue-700 border border-blue-200"
              trend="up"
              trendValue={`+${analytics?.usage_stats?.recent_recipes || 0}`}
              isUpdated={lastUpdated && (new Date() - lastUpdated) < 300000}
            />
          </div>
        </div>

        {/* Environmental Impact Section */}
        {analytics && (
          <div className="dashboard-section">
            <div className="section-header">
              <h2 className="section-title">
                <span className="section-icon">🌍</span>
                Environmental Impact
              </h2>
            </div>
            
            <div className="impact-grid">
              <div className="impact-card">
                <div className="impact-header">
                  <div className="impact-icon">
                    <span>🌍</span>
                  </div>
                  <h3 className="impact-title">Carbon Footprint Reduction</h3>
                </div>
                <div className="impact-metrics">
                  <div className="impact-metric">
                    <div className="impact-metric-value">{analytics.waste_reduction.co2_saved_kg}kg</div>
                    <div className="impact-metric-label">CO₂ Saved</div>
                  </div>
                  <div className="impact-metric">
                    <div className="impact-metric-value">{analytics.waste_reduction.water_saved_liters}L</div>
                    <div className="impact-metric-label">Water Saved</div>
                  </div>
                  <div className="impact-metric">
                    <div className="impact-metric-value">{analytics.environmental_impact.trees_equivalent}</div>
                    <div className="impact-metric-label">Trees Equivalent</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Recipe Statistics Section */}
        {recipeStats && (
          <div className="dashboard-section">
            <div className="section-header">
              <h2 className="section-title">
                <span className="section-icon">👨‍🍳</span>
                Recipe Generation Stats
              </h2>
            </div>
            
            <div className="impact-grid">
              <div className="impact-card impact-card-blue">
                <div className="impact-header">
                  <div className="impact-icon impact-icon-blue">
                    <span>👨‍🍳</span>
                  </div>
                  <h3 className="impact-title">AI Recipe Performance</h3>
                </div>
                <div className="impact-metrics">
                  <div className="impact-metric">
                    <div className="impact-metric-value">{recipeStats.scan_statistics.success_rate_percent}%</div>
                    <div className="impact-metric-label">Success Rate</div>
                  </div>
                  <div className="impact-metric">
                    <div className="impact-metric-value">{recipeStats.scan_statistics.average_recipes_per_scan}</div>
                    <div className="impact-metric-label">Avg Recipes per Scan</div>
                  </div>
                  <div className="impact-metric">
                    <div className="impact-metric-value">{recipeStats.usage_stats.total_scans || 0}</div>
                    <div className="impact-metric-label">Total Scans</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tips Section */}
        <div className="dashboard-section">
          <div className="section-header">
            <h2 className="section-title">
              <span className="section-icon">💡</span>
              Tips to Reduce Food Waste
            </h2>
          </div>
          
          <div className="tips-section">
            <div className="tips-header">
              <div className="tips-icon">
                <span>💡</span>
              </div>
              <h3 className="tips-title">Smart Food Management</h3>
            </div>
            <div className="tips-grid">
              {[
                'Scan items when you buy them to track inventory',
                'Use recipe generator for expiring items',
                'Check ingredients regularly and plan meals',
                'Store food properly to extend shelf life',
                'Compost food scraps when possible',
                'Share excess food with neighbors or community'
              ].map((tip, index) => (
                <div key={index} className="tip-item">
                  <div className="tip-dot"></div>
                  <span className="tip-text">{tip}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;