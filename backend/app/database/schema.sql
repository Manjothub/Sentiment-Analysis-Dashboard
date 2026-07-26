-- ============================================================================
-- Sentiment Analysis Dashboard - Database Schema
-- ============================================================================
-- This SQL file defines the PostgreSQL database schema for the Sentiment
-- Analysis Dashboard. It includes tables for products, reviews, and
-- sentiment analysis results with proper indexes, constraints, and
-- performance optimizations.
--
-- Database: sentiment_dashboard
-- PostgreSQL Version: 15+
-- ============================================================================

-- Create database (run separately if needed)
-- CREATE DATABASE sentiment_dashboard;

-- ============================================================================
-- Table: products
-- Purpose: Stores product information from Amazon reviews
-- ============================================================================
CREATE TABLE IF NOT EXISTS products (
    -- Primary key
    product_id      VARCHAR(50) PRIMARY KEY,
    
    -- Product identifiers
    asin            VARCHAR(20) UNIQUE,                    -- Amazon Standard Identification Number
    product_name    VARCHAR(500) NOT NULL,                  -- Product title/name
    brand           VARCHAR(200),                           -- Brand name
    category        VARCHAR(200),                           -- Product category
    price           DECIMAL(10, 2),                         -- Product price (if available)
    
    -- Metadata
    is_competitor   BOOLEAN DEFAULT FALSE,                  -- Flag for competitor products
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(), -- Record creation timestamp
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(), -- Last update timestamp
    
    -- Constraints
    CONSTRAINT check_price_positive CHECK (price IS NULL OR price > 0)
);

-- Indexes for products table
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);
CREATE INDEX IF NOT EXISTS idx_products_is_competitor ON products(is_competitor);
CREATE INDEX IF NOT EXISTS idx_products_created_at ON products(created_at);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_products_updated_at
    BEFORE UPDATE ON products
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================================================
-- Table: reviews
-- Purpose: Stores product reviews with raw text and metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS reviews (
    -- Primary key
    review_id       BIGSERIAL PRIMARY KEY,
    
    -- Foreign key to products
    product_id      VARCHAR(50) NOT NULL,
    
    -- Reviewer information
    reviewer_name   VARCHAR(200),                           -- Reviewer username/ID
    reviewer_id     VARCHAR(100),                           -- Internal reviewer identifier
    
    -- Review content
    review_text     TEXT NOT NULL,                          -- Original review text
    cleaned_text    TEXT,                                   -- Preprocessed/cleaned text
    summary         VARCHAR(500),                           -- Review summary/title
    
    -- Ratings and metadata
    raw_rating      SMALLINT,                               -- Original rating (1-5)
    review_date     TIMESTAMP WITH TIME ZONE,               -- Date of review
    verified_purchase BOOLEAN DEFAULT FALSE,                -- Verified purchase flag
    helpful_votes   INTEGER DEFAULT 0,                      -- Number of helpful votes
    total_votes     INTEGER DEFAULT 0,                      -- Total votes received
    
    -- Processing metadata
    source          VARCHAR(20) DEFAULT 'amazon',           -- Data source (amazon/twitter)
    ingested_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(), -- When was this record ingested
    
    -- Constraints
    CONSTRAINT fk_reviews_product
        FOREIGN KEY (product_id)
        REFERENCES products(product_id)
        ON DELETE CASCADE,                                  -- Delete reviews if product is deleted
    CONSTRAINT check_rating_range CHECK (raw_rating IS NULL OR (raw_rating >= 1 AND raw_rating <= 5)),
    CONSTRAINT check_review_length CHECK (LENGTH(review_text) >= 1)
);

-- Indexes for reviews table
CREATE INDEX IF NOT EXISTS idx_reviews_product_id ON reviews(product_id);
CREATE INDEX IF NOT EXISTS idx_reviews_raw_rating ON reviews(raw_rating);
CREATE INDEX IF NOT EXISTS idx_reviews_review_date ON reviews(review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_verified_purchase ON reviews(verified_purchase);
CREATE INDEX IF NOT EXISTS idx_reviews_source ON reviews(source);
CREATE INDEX IF NOT EXISTS idx_reviews_ingested_at ON reviews(ingested_at);
CREATE INDEX IF NOT EXISTS idx_reviews_helpful_votes ON reviews(helpful_votes DESC);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_reviews_product_rating ON reviews(product_id, raw_rating);
CREATE INDEX IF NOT EXISTS idx_reviews_product_date ON reviews(product_id, review_date DESC);

-- Full-text search index for review text
CREATE INDEX IF NOT EXISTS idx_reviews_text_search
    ON reviews
    USING gin(to_tsvector('english', review_text));


-- ============================================================================
-- Table: sentiment_results
-- Purpose: Stores sentiment analysis predictions for reviews
-- ============================================================================
CREATE TABLE IF NOT EXISTS sentiment_results (
    -- Primary key
    result_id       BIGSERIAL PRIMARY KEY,
    
    -- Foreign key to reviews
    review_id       BIGINT NOT NULL UNIQUE,                 -- One sentiment result per review
    
    -- Sentiment predictions
    predicted_sentiment VARCHAR(20) NOT NULL,                -- positive/negative/neutral
    positive_score  DOUBLE PRECISION DEFAULT 0.0,           -- Confidence score for positive
    negative_score  DOUBLE PRECISION DEFAULT 0.0,           -- Confidence score for negative
    neutral_score   DOUBLE PRECISION DEFAULT 0.0,           -- Confidence score for neutral
    confidence_score DOUBLE PRECISION DEFAULT 0.0,           -- Overall confidence
    
    -- Detailed analysis
    emotion         VARCHAR(50),                            -- Detected emotion (joy, anger, etc.)
    aspects         JSONB DEFAULT '{}'::jsonb,              -- Extracted aspects and their sentiments
    aspect_scores   JSONB DEFAULT '{}'::jsonb,              -- Numerical scores for each aspect
    
    -- Model metadata
    model_version   VARCHAR(50),                            -- Version of model used
    model_name      VARCHAR(100) DEFAULT 'distilbert-sentiment', -- Model identifier
    inference_time_ms INTEGER,                              -- Inference time in milliseconds
    
    -- Timestamps
    analyzed_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(), -- When analysis was performed
    
    -- Constraints
    CONSTRAINT fk_sentiment_review
        FOREIGN KEY (review_id)
        REFERENCES reviews(review_id)
        ON DELETE CASCADE,                                  -- Delete results if review is deleted
    CONSTRAINT check_predicted_sentiment
        CHECK (predicted_sentiment IN ('positive', 'negative', 'neutral')),
    CONSTRAINT check_score_range
        CHECK (positive_score >= 0 AND positive_score <= 1
           AND negative_score >= 0 AND negative_score <= 1
           AND neutral_score >= 0 AND neutral_score <= 1)
);

-- Indexes for sentiment_results table
CREATE INDEX IF NOT EXISTS idx_sentiment_review_id ON sentiment_results(review_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_predicted ON sentiment_results(predicted_sentiment);
CREATE INDEX IF NOT EXISTS idx_sentiment_confidence ON sentiment_results(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_model_version ON sentiment_results(model_version);
CREATE INDEX IF NOT EXISTS idx_sentiment_analyzed_at ON sentiment_results(analyzed_at DESC);

-- Composite indexes
CREATE INDEX IF NOT EXISTS idx_sentiment_review_analysis
    ON sentiment_results(review_id, predicted_sentiment, analyzed_at);

-- GIN index for JSONB aspects queries
CREATE INDEX IF NOT EXISTS idx_sentiment_aspects ON sentiment_results USING gin(aspects);
CREATE INDEX IF NOT EXISTS idx_sentiment_aspect_scores ON sentiment_results USING gin(aspect_scores);


-- ============================================================================
-- Table: model_versions
-- Purpose: Track model versions and their metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_versions (
    version_id      SERIAL PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL,
    version         VARCHAR(50) NOT NULL,
    accuracy        DOUBLE PRECISION,
    f1_score        DOUBLE PRECISION,
    precision       DOUBLE PRECISION,
    recall          DOUBLE PRECISION,
    training_date   TIMESTAMP WITH TIME ZONE,
    parameters      JSONB DEFAULT '{}'::jsonb,
    notes           TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_versions_name ON model_versions(model_name);
CREATE INDEX IF NOT EXISTS idx_model_versions_created ON model_versions(created_at DESC);


-- ============================================================================
-- Views for common analytics queries
-- ============================================================================

-- View: Product sentiment summary
CREATE OR REPLACE VIEW v_product_sentiment AS
SELECT
    p.product_id,
    p.product_name,
    p.category,
    COUNT(r.review_id) AS total_reviews,
    COUNT(CASE WHEN r.raw_rating >= 4 THEN 1 END) AS positive_reviews,
    COUNT(CASE WHEN r.raw_rating = 3 THEN 1 END) AS neutral_reviews,
    COUNT(CASE WHEN r.raw_rating <= 2 THEN 1 END) AS negative_reviews,
    AVG(r.raw_rating) AS avg_rating,
    COUNT(CASE WHEN s.predicted_sentiment = 'positive' THEN 1 END) AS predicted_positive,
    COUNT(CASE WHEN s.predicted_sentiment = 'negative' THEN 1 END) AS predicted_negative,
    COUNT(CASE WHEN s.predicted_sentiment = 'neutral' THEN 1 END) AS predicted_neutral,
    AVG(s.positive_score) AS avg_positive_score,
    AVG(s.negative_score) AS avg_negative_score,
    AVG(s.confidence_score) AS avg_confidence
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
LEFT JOIN sentiment_results s ON r.review_id = s.review_id
GROUP BY p.product_id, p.product_name, p.category;

-- View: Sentiment trend over time
CREATE OR REPLACE VIEW v_sentiment_trend AS
SELECT
    DATE_TRUNC('day', r.review_date) AS review_day,
    p.category,
    COUNT(*) AS total_reviews,
    AVG(r.raw_rating) AS avg_rating,
    COUNT(CASE WHEN s.predicted_sentiment = 'positive' THEN 1 END) AS positive_count,
    COUNT(CASE WHEN s.predicted_sentiment = 'negative' THEN 1 END) AS negative_count,
    COUNT(CASE WHEN s.predicted_sentiment = 'neutral' THEN 1 END) AS neutral_count
FROM reviews r
JOIN sentiment_results s ON r.review_id = s.review_id
JOIN products p ON r.product_id = p.product_id
WHERE r.review_date IS NOT NULL
GROUP BY DATE_TRUNC('day', r.review_date), p.category
ORDER BY review_day DESC;


-- ============================================================================
-- Permissions
-- ============================================================================
-- Grant appropriate permissions (adjust based on your security requirements)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO sentiment_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO sentiment_app;
