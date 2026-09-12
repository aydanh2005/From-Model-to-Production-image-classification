CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    source_file TEXT,
    row_index INT,
    predicted_class INT,
    predicted_label TEXT,
    confidence FLOAT,
    processed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (source_file, row_index)
);
