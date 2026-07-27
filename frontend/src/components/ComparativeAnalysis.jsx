import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { comparativeAPI } from '../services/api';

const COLORS = { positive: '#10b981', negative: '#ef4444', neutral: '#f59e0b' };

const ComparativeAnalysis = ({ comparison = null, aspectComparison = {} }) => {
  const [productId, setProductId] = useState('');
  const [competitorId, setCompetitorId] = useState('');
  const [compData, setCompData] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchComparison = async () => {
    if (!productId) return;
    setLoading(true);
    try {
      const res = await comparativeAPI.compare({ product_id: productId });
      setCompData(res.data);
    } catch (err) {
      console.error('Comparison fetch failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const hasData = comparison && comparison.product && comparison.competitors;
  const source = compData || comparison;

  const overallChart = () => {
    if (!source || !source.product) return null;
    const product = source.product;
    const competitors = source.competitors || [];
    const sentimentDist = product.sentiment_distribution || {};
    const total = sentimentDist.positive + sentimentDist.negative + sentimentDist.neutral || 1;

    const entities = [
      { name: product.product_name || product.product_id, data: {
        positive_pct: (sentimentDist.positive / total) * 100,
        negative_pct: (sentimentDist.negative / total) * 100,
        neutral_pct: (sentimentDist.neutral / total) * 100,
      }},
      ...competitors.map((c) => {
        const dist = c.sentiment_distribution || {};
        const t = dist.positive + dist.negative + dist.neutral || 1;
        return { name: c.product_name || c.product_id, data: {
          positive_pct: (dist.positive / t) * 100,
          negative_pct: (dist.negative / t) * 100,
          neutral_pct: (dist.neutral / t) * 100,
        }};
      }),
    ];

    const data = [
      { x: entities.map((e) => e.name), y: entities.map((e) => e.data.positive_pct), name: 'Positive', type: 'bar', marker: { color: COLORS.positive } },
      { x: entities.map((e) => e.name), y: entities.map((e) => e.data.negative_pct), name: 'Negative', type: 'bar', marker: { color: COLORS.negative } },
      { x: entities.map((e) => e.name), y: entities.map((e) => e.data.neutral_pct), name: 'Neutral', type: 'bar', marker: { color: COLORS.neutral } },
    ];

    const layout = {
      height: 280, barmode: 'group', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#cbd5e1', family: 'Plus Jakarta Sans, sans-serif' },
      margin: { t: 20, b: 60, l: 50, r: 20 },
      xaxis: { gridcolor: 'rgba(255,255,255,0.06)' },
      yaxis: { gridcolor: 'rgba(255,255,255,0.06)', title: 'Percentage (%)' },
      legend: { orientation: 'h', y: 1.15, font: { color: '#f8fafc' } },
    };

    return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} />;
  };

  const aspectChart = () => {
    const aspects = Object.keys(aspectComparison);
    if (aspects.length === 0) return null;

    const ASPECT_LABELS = {
      quality: 'Quality', shipping: 'Shipping',
      customer_service: 'Service', value: 'Value', usability: 'Usability',
    };

    const data = [
      {
        x: aspects.map((a) => ASPECT_LABELS[a] || a),
        y: aspects.map((a) => aspectComparison[a]?.product?.positive_pct || 0),
        name: 'Product', type: 'bar', marker: { color: '#6366f1' },
      },
      {
        x: aspects.map((a) => ASPECT_LABELS[a] || a),
        y: aspects.map((a) => aspectComparison[a]?.competitor?.positive_pct || 0),
        name: 'Competitor', type: 'bar', marker: { color: '#f59e0b' },
      },
    ];

    const layout = {
      height: 280, barmode: 'group', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#cbd5e1', family: 'Plus Jakarta Sans, sans-serif' },
      margin: { t: 20, b: 60, l: 50, r: 20 },
      xaxis: { gridcolor: 'rgba(255,255,255,0.06)' },
      yaxis: { gridcolor: 'rgba(255,255,255,0.06)', title: 'Positive %' },
      legend: { orientation: 'h', y: 1.15, font: { color: '#f8fafc' } },
    };

    return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} />;
  };

  return (
    <div>
      <div className="row g-3 mb-3">
        <div className="col-md-4">
          <input type="text" className="form-control" placeholder="Product ID" value={productId} onChange={(e) => setProductId(e.target.value)} />
        </div>
        <div className="col-md-4">
          <input type="text" className="form-control" placeholder="Competitor ID (optional)" value={competitorId} onChange={(e) => setCompetitorId(e.target.value)} />
        </div>
        <div className="col-md-4">
          <button className="btn btn-primary w-100" onClick={fetchComparison} disabled={loading}>
            {loading ? 'Comparing...' : 'Compare'}
          </button>
        </div>
      </div>

      {hasData ? (
        <>
          <div className="mb-3">{overallChart()}</div>
          <div>{aspectChart()}</div>
        </>
      ) : compData ? (
        <div className="text-center text-muted py-5">
          <p>No comparison data available for the selected product</p>
        </div>
      ) : (
        <div className="text-center text-muted py-5">
          <p>Enter a product ID and click Compare to analyze sentiment across products and competitors</p>
        </div>
      )}
    </div>
  );
};

export default ComparativeAnalysis;
