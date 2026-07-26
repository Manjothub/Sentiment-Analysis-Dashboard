import Plot from 'react-plotly.js';

const COLORS = { positive: '#2ecc71', negative: '#e74c3c', neutral: '#f39c12' };

const SentimentChart = ({ trendData = [] }) => {
  const dates = trendData.map((d) => d.date);
  const data = [
    {
      x: dates,
      y: trendData.map((d) => d.positive),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Positive',
      line: { color: COLORS.positive, width: 2 },
      marker: { size: 4 },
      fill: 'tozeroy',
      fillcolor: 'rgba(46, 204, 113, 0.1)',
    },
    {
      x: dates,
      y: trendData.map((d) => d.negative),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Negative',
      line: { color: COLORS.negative, width: 2 },
      marker: { size: 4 },
      fill: 'tozeroy',
      fillcolor: 'rgba(231, 76, 60, 0.1)',
    },
    {
      x: dates,
      y: trendData.map((d) => d.neutral),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Neutral',
      line: { color: COLORS.neutral, width: 2 },
      marker: { size: 4 },
      fill: 'tozeroy',
      fillcolor: 'rgba(243, 156, 18, 0.1)',
    },
  ];

  const layout = {
    height: 300,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e0e0e0', family: 'Segoe UI, sans-serif' },
    margin: { t: 20, b: 40, l: 50, r: 20 },
    xaxis: { gridcolor: '#333', title: 'Date' },
    yaxis: { gridcolor: '#333', title: 'Count' },
    legend: { orientation: 'h', y: 1.1 },
    hovermode: 'x unified',
  };

  return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} />;
};

export default SentimentChart;
