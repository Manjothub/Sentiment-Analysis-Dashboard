import Plot from 'react-plotly.js';

const SentimentGauge = ({ positive = 0, negative = 0, neutral = 0 }) => {
  const total = (positive + negative + neutral) || 1;

  const data = [
    {
      type: 'indicator',
      mode: 'gauge+number+delta',
      value: (positive / total) * 100,
      title: { text: 'Positive Sentiment', font: { size: 16, color: '#f8fafc', family: 'Plus Jakarta Sans' } },
      delta: { reference: 50, increasing: { color: '#10b981' } },
      gauge: {
        axis: { range: [0, 100], tickwidth: 1, tickcolor: '#64748b' },
        bar: { color: '#10b981' },
        steps: [
          { range: [0, 33], color: 'rgba(239, 68, 68, 0.85)' },
          { range: [33, 66], color: 'rgba(245, 158, 11, 0.85)' },
          { range: [66, 100], color: 'rgba(16, 185, 129, 0.85)' },
        ],
        threshold: {
          line: { color: '#6366f1', width: 4 },
          thickness: 0.75,
          value: 50,
        },
      },
      number: { suffix: '%', font: { size: 26, color: '#f8fafc', family: 'Plus Jakarta Sans' } },
    },
  ];

  const layout = {
    height: 220,
    margin: { t: 30, b: 10, l: 30, r: 30 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', family: 'Plus Jakarta Sans, sans-serif' },
  };

  return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} />;
};

export default SentimentGauge;
