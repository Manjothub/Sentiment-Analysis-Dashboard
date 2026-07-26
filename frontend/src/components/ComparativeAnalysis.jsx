import Plot from 'react-plotly.js';

const ComparativeAnalysis = ({ comparison = null, aspectComparison = {} }) => {
  const hasData = comparison && comparison.product_sentiment;

  const overallChart = () => {
    if (!hasData) return null;
    const { product_sentiment: product, competitors = {} } = comparison;

    const entities = [
      { name: comparison.product_id || 'Product', data: product },
      ...Object.entries(competitors).map(([id, comp]) => ({
        name: id,
        data: comp.product,
      })),
    ];

    const data = [
      {
        x: entities.map((e) => e.name),
        y: entities.map((e) => e.data.positive_pct || 0),
        name: 'Positive',
        type: 'bar',
        marker: { color: '#2ecc71' },
      },
      {
        x: entities.map((e) => e.name),
        y: entities.map((e) => e.data.negative_pct || 0),
        name: 'Negative',
        type: 'bar',
        marker: { color: '#e74c3c' },
      },
      {
        x: entities.map((e) => e.name),
        y: entities.map((e) => e.data.neutral_pct || 0),
        name: 'Neutral',
        type: 'bar',
        marker: { color: '#f39c12' },
      },
    ];

    const layout = {
      height: 280,
      barmode: 'group',
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#e0e0e0', family: 'Segoe UI, sans-serif' },
      margin: { t: 20, b: 60, l: 50, r: 20 },
      xaxis: { gridcolor: '#333' },
      yaxis: { gridcolor: '#333', title: 'Percentage (%)' },
      legend: { orientation: 'h', y: 1.1 },
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
        name: 'Product',
        type: 'bar',
        marker: { color: '#3498db' },
      },
      {
        x: aspects.map((a) => ASPECT_LABELS[a] || a),
        y: aspects.map((a) => aspectComparison[a]?.competitor?.positive_pct || 0),
        name: 'Competitor',
        type: 'bar',
        marker: { color: '#e67e22' },
      },
    ];

    const layout = {
      height: 280,
      barmode: 'group',
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#e0e0e0', family: 'Segoe UI, sans-serif' },
      margin: { t: 20, b: 60, l: 50, r: 20 },
      xaxis: { gridcolor: '#333' },
      yaxis: { gridcolor: '#333', title: 'Positive %' },
      legend: { orientation: 'h', y: 1.1 },
    };

    return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} />;
  };

  return (
    <div>
      {hasData ? (
        <>
          <div className="mb-3">{overallChart()}</div>
          <div>{aspectChart()}</div>
        </>
      ) : (
        <div className="text-center text-muted py-5">
          <p>Select a product and competitor to compare</p>
        </div>
      )}
    </div>
  );
};

export default ComparativeAnalysis;
