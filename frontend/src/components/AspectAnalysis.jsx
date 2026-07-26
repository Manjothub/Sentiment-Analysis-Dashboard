import Plot from 'react-plotly.js';

const ASPECT_LABELS = {
  quality: 'Product Quality',
  shipping: 'Shipping',
  customer_service: 'Customer Service',
  value: 'Value for Money',
  usability: 'Usability',
};

const ASPECT_COLORS = { positive: '#2ecc71', negative: '#e74c3c', neutral: '#f39c12' };

const AspectAnalysis = ({ aspects = {} }) => {
  const aspectNames = Object.keys(aspects);

  const data = [
    {
      y: aspectNames.map((k) => ASPECT_LABELS[k] || k),
      x: aspectNames.map((k) => aspects[k].positive_pct || 0),
      name: 'Positive',
      type: 'bar',
      orientation: 'h',
      marker: { color: ASPECT_COLORS.positive },
      text: aspectNames.map((k) => `${aspects[k].mentions || 0} mentions`),
      textposition: 'outside',
    },
    {
      y: aspectNames.map((k) => ASPECT_LABELS[k] || k),
      x: aspectNames.map((k) => aspects[k].negative_pct || 0),
      name: 'Negative',
      type: 'bar',
      orientation: 'h',
      marker: { color: ASPECT_COLORS.negative },
      textposition: 'outside',
    },
  ];

  const layout = {
    height: 300,
    barmode: 'group',
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e0e0e0', family: 'Segoe UI, sans-serif' },
    margin: { t: 20, b: 30, l: 140, r: 80 },
    xaxis: { gridcolor: '#333', title: 'Percentage (%)', range: [0, 100] },
    yaxis: { gridcolor: '#333' },
    legend: { orientation: 'h', y: 1.1 },
    hovermode: 'y unified',
  };

  if (aspectNames.length === 0) {
    return (
      <div className="text-center text-muted py-5">
        <p>No aspect data available yet</p>
      </div>
    );
  }

  return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} />;
};

export default AspectAnalysis;
