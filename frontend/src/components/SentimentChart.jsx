import Plot from 'react-plotly.js';

const COLORS = { positive: '#10b981', negative: '#ef4444', neutral: '#f59e0b' };

const SentimentChart = ({ trendData = [] }) => {
  const dates = trendData.map((d) => d.date);
  const data = [
    {
      x: dates,
      y: trendData.map((d) => d.positive),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Positive',
      line: { color: COLORS.positive, width: 2.5, shape: 'spline' },
      marker: { size: 5 },
      fill: 'tozeroy',
      fillcolor: 'rgba(16, 185, 129, 0.12)',
    },
    {
      x: dates,
      y: trendData.map((d) => d.negative),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Negative',
      line: { color: COLORS.negative, width: 2.5, shape: 'spline' },
      marker: { size: 5 },
      fill: 'tozeroy',
      fillcolor: 'rgba(239, 68, 68, 0.12)',
    },
    {
      x: dates,
      y: trendData.map((d) => d.neutral),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Neutral',
      line: { color: COLORS.neutral, width: 2.5, shape: 'spline' },
      marker: { size: 5 },
      fill: 'tozeroy',
      fillcolor: 'rgba(245, 158, 11, 0.12)',
    },
  ];

  const layout = {
    height: 280,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', family: 'Plus Jakarta Sans, sans-serif' },
    margin: { t: 20, b: 40, l: 45, r: 20 },
    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', title: 'Date' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', title: 'Count' },
    legend: { orientation: 'h', y: 1.15, font: { color: '#f8fafc' } },
    hovermode: 'x unified',
  };

  return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} />;
};

export default SentimentChart;
