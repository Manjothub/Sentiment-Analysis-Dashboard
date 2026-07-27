import Plot from 'react-plotly.js';

const getSentimentColor = (sentiment) => {
  switch (sentiment) {
    case 'positive': return '#10b981';
    case 'negative': return '#ef4444';
    default: return '#f59e0b';
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
    height: 360,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#cbd5e1', family: 'Plus Jakarta Sans, sans-serif' },
    margin: { t: 20, b: 40, l: 120, r: 80 },
    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', title: 'Frequency' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', automargin: true },
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
