import re
import json
import numpy as np
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

ASPECT_KEYWORDS = {
    'quality': [
        'quality', 'durable', 'sturdy', 'well-made', 'poor quality',
        'cheap', 'broken', 'defective', 'material', 'build',
        'solid', 'reliable', 'craftsmanship', 'finish'
    ],
    'shipping': [
        'shipping', 'delivery', 'arrived', 'package', 'shipment',
        'fast shipping', 'slow shipping', 'shipping time',
        'delivered', 'tracking', 'packaging', 'arrival'
    ],
    'customer_service': [
        'customer service', 'support', 'refund', 'return',
        'helpful', 'response', 'contacted', 'representative',
        'replacement', 'complaint', 'warranty', 'assistance'
    ],
    'value': [
        'price', 'worth', 'expensive', 'cheap', 'overpriced',
        'value', 'bargain', 'cost', 'affordable', 'reasonable',
        'pricey', 'deal', 'paid', 'spent'
    ],
    'usability': [
        'easy', 'difficult', 'setup', 'install', 'use',
        'user-friendly', 'intuitive', 'confusing', 'complicated',
        'instruction', 'manual', 'guide', 'interface'
    ]
}

ASPECT_SENTIMENT_KEYWORDS = {
    'positive': [
        'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'perfect', 'best', 'awesome', 'outstanding',
        'satisfied', 'happy', 'impressed', 'recommend', 'superb'
    ],
    'negative': [
        'terrible', 'awful', 'horrible', 'worst', 'disappointed',
        'frustrating', 'useless', 'waste', 'regret', 'poor',
        'bad', 'dreadful', 'unacceptable', 'defective', 'hate'
    ]
}


class SentimentAnalyzer:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.load_model()

    def load_model(self):
        try:
            if self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'distilbert-base-uncased-finetuned-sst-2-english'
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    'distilbert-base-uncased-finetuned-sst-2-english'
                )
            self.classifier = pipeline(
                'sentiment-analysis',
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=True,
                max_length=128
            )
            return True
        except Exception as e:
            print(f"Model loading error: {e}")
            return False

    def analyze_sentiment(self, text):
        if not text or not text.strip():
            return {
                'label': 'neutral',
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 1.0,
                'aspects': {}
            }
        try:
            result = self.classifier(text[:512])
            scores = {r['label'].lower(): r['score'] for r in result[0]}
            pos_score = scores.get('positive', scores.get('pos', 0.0))
            neg_score = scores.get('negative', scores.get('neg', 0.0))
            neutral_score = max(0.0, 1.0 - pos_score - neg_score)

            if pos_score > neg_score and pos_score > 0.4:
                label = 'positive'
            elif neg_score > pos_score and neg_score > 0.4:
                label = 'negative'
            else:
                label = 'neutral'

            aspects = self.extract_aspects(text)

            return {
                'label': label,
                'positive_score': round(pos_score, 4),
                'negative_score': round(neg_score, 4),
                'neutral_score': round(neutral_score, 4),
                'aspects': aspects
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {
                'label': 'neutral',
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 1.0,
                'aspects': {}
            }

    def extract_aspects(self, text):
        text_lower = text.lower()
        aspects = {}
        for aspect, keywords in ASPECT_KEYWORDS.items():
            score = 0
            matched_keywords = []
            for kw in keywords:
                if kw in text_lower:
                    score += 1
                    matched_keywords.append(kw)

            if score > 0:
                sentiment = self._get_aspect_sentiment(text_lower, matched_keywords)
                aspects[aspect] = {
                    'score': score,
                    'sentiment': sentiment,
                    'keywords': matched_keywords
                }
        return aspects

    def _get_aspect_sentiment(self, text, matched_keywords):
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = [
            s for s in sentences
            if any(kw in s for kw in matched_keywords)
        ]
        combined = ' '.join(relevant_sentences)
        pos_count = sum(
            1 for kw in ASPECT_SENTIMENT_KEYWORDS['positive'] if kw in combined
        )
        neg_count = sum(
            1 for kw in ASPECT_SENTIMENT_KEYWORDS['negative'] if kw in combined
        )
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'

    def extract_trending_topics(self, texts, top_n=10):
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            all_words.extend([w for w in words if w not in STOP_WORDS])

        word_freq = Counter(all_words)
        return word_freq.most_common(top_n)

    def detect_sentiment_spike(self, recent_sentiments, window_size=10, threshold=0.3):
        if len(recent_sentiments) < window_size:
            return None
        recent = recent_sentiments[-window_size:]
        neg_count = sum(1 for s in recent if s == 'negative')
        neg_ratio = neg_count / window_size

        if neg_ratio >= threshold:
            return {
                'spike_detected': True,
                'negative_ratio': neg_ratio,
                'threshold': threshold,
                'window_size': window_size
            }
        return None

    def comparative_analysis(self, product_texts, competitor_texts):
        product_scores = [self.analyze_sentiment(t) for t in product_texts]
        competitor_scores = [self.analyze_sentiment(t) for t in competitor_texts]

        def aggregate(scores):
            labels = [s['label'] for s in scores]
            pos = labels.count('positive') / max(len(labels), 1)
            neg = labels.count('negative') / max(len(labels), 1)
            neu = labels.count('neutral') / max(len(labels), 1)
            avg_pos = np.mean([s['positive_score'] for s in scores]) if scores else 0
            avg_neg = np.mean([s['negative_score'] for s in scores]) if scores else 0
            return {
                'positive_pct': round(pos * 100, 1),
                'negative_pct': round(neg * 100, 1),
                'neutral_pct': round(neu * 100, 1),
                'avg_positive_score': round(float(avg_pos), 4),
                'avg_negative_score': round(float(avg_neg), 4),
                'total_reviews': len(scores)
            }

        return {
            'product': aggregate(product_scores),
            'competitor': aggregate(competitor_scores)
        }
