import Plot from 'react-plotly.js';

const getSentimentColor = (sentiment) => {
  switch (sentiment) {
    case 'positive': return '#2ecc71';
    case 'negative': return '#e74c3c';
    default: return '#f39c12';
  }
};

const TrendingTopics = ({ topics = [] }) => {
  const topTopics = topics.slice(0, 15);

  const data = [
    {
      x: topTopics.map((t) => t.frequency),
      y: topTopics.map((t) => t.keyword).reverse(),
      type: 'bar',
      orientation: 'h',
      marker: {
        color: topTopics.map((t) => getSentimentColor(t.dominant_sentiment)).reverse(),
      },
      text: topTopics.map((t) => t.dominant_sentiment).reverse(),
      textposition: 'outside',
      hovertemplate: '%{y}: %{x} mentions<br>Sentiment: %{text}<extra></extra>',
    },
  ];

  const layout = {
    height: 350,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e0e0e0', family: 'Segoe UI, sans-serif' },
    margin: { t: 20, b: 40, l: 120, r: 60 },
    xaxis: { gridcolor: '#333', title: 'Frequency' },
    yaxis: { gridcolor: '#333', automargin: true },
    hovermode: 'y',
  };

  if (topTopics.length === 0) {
    return (
      <div className="text-center text-muted py-5">
        <p>No trending topics data yet</p>
      </div>
    );
  }

  return <Plot data={data} layout={layout} config={{ responsive: true, displayModeBar: false }} />;
};

export default TrendingTopics;
