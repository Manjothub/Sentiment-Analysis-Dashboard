import Plot from 'react-plotly.js';

const SentimentGauge = ({ positive = 0, negative = 0, neutral = 0 }) => {
  const total = positive + negative + neutral || 1;

  const data = [
    {
      type: 'indicator',
      mode: 'gauge+number+delta',
      value: (positive / total) * 100,
      title: { text: 'Positive Sentiment', font: { size: 16 } },
      delta: { reference: 50, increasing: { color: '#2ecc71' } },
      gauge: {
        axis: { range: [0, 100], tickwidth: 1 },
        bar: { color: '#2ecc71' },
        steps: [
          { range: [0, 33], color: '#e74c3c' },
          { range: [33, 66], color: '#f39c12' },
          { range: [66, 100], color: '#2ecc71' },
        ],
        threshold: {
          line: { color: 'red', width: 4 },
          thickness: 0.75,
          value: 50,
        },
      },
      number: { suffix: '%', font: { size: 24 } },
    },
  ];

  const layout = {
    height: 250,
    margin: { t: 40, b: 20, l: 20, r: 20 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e0e0e0', family: 'Segoe UI, sans-serif' },
  };

  return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} />;
};

export default SentimentGauge;
