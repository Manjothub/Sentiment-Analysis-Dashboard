import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_amazon_data(filepath, sample_size=50000):
    print(f"Loading Amazon reviews from {filepath}...")
    df = pd.read_csv(filepath, nrows=sample_size)
    df = df.dropna(subset=['Text', 'Score'])
    df['label'] = df['Score'].apply(
        lambda x: 2 if x >= 4 else (1 if x == 3 else 0)
    )
    df['text'] = df['Text'].str[:512]
    print(f"Loaded {len(df)} Amazon samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df[['text', 'label']]


def load_twitter_data(filepath, sample_size=100000):
    print(f"Loading Twitter sentiment from {filepath}...")
    df = pd.read_csv(
        filepath,
        encoding='latin-1',
        header=None,
        names=['sentiment', 'id', 'date', 'query', 'user', 'text'],
        nrows=sample_size
    )
    df = df.dropna(subset=['text'])
    df['label'] = df['sentiment'].apply(lambda x: 0 if x == 0 else 2)
    df['text'] = df['text'].str[:512]
    print(f"Loaded {len(df)} Twitter samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df[['text', 'label']]


def compute_metrics(eval_pred):
    accuracy = evaluate.load('accuracy')
    f1 = evaluate.load('f1')
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        'accuracy': accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        'f1': f1.compute(predictions=predictions, references=labels, average='weighted')['f1'],
        'precision': precision.compute(predictions=predictions, references=labels, average='weighted')['precision'],
        'recall': recall.compute(predictions=predictions, references=labels, average='weighted')['recall'],
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune DistilBERT for sentiment analysis')
    parser.add_argument('--amazon_path', type=str,
                        default='dataset/amazon-reviews/Reviews.csv')
    parser.add_argument('--twitter_path', type=str,
                        default='dataset/twitter-sentiment/training.1600000.processed.noemoticon.csv')
    parser.add_argument('--output_dir', type=str,
                        default='saved_model/distilbert-sentiment')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_samples', type=int, default=150000)
    parser.add_argument('--amazon_samples', type=int, default=50000)
    parser.add_argument('--twitter_samples', type=int, default=100000)

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    amazon_path = os.path.join(base_dir, args.amazon_path)
    twitter_path = os.path.join(base_dir, args.twitter_path)

    amazon_df = load_amazon_data(amazon_path, args.amazon_samples)
    twitter_df = load_twitter_data(twitter_path, args.twitter_samples)

    total_samples = min(len(amazon_df) + len(twitter_df), args.max_samples)
    df = pd.concat([amazon_df, twitter_df], ignore_index=True)
    if len(df) > total_samples:
        df = df.sample(n=total_samples, random_state=42)

    print(f"Total training samples: {len(df)}")

    train_df, eval_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f"Train: {len(train_df)}, Eval: {len(eval_df)}")

    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'label']])

    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: 'negative', 1: 'neutral', 2: 'positive'},
        label2id={'negative': 0, 'neutral': 1, 'positive': 2}
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_dir='./logs',
        logging_steps=500,
        save_total_limit=2,
        fp16=False,
        report_to=['tensorboard'],
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

    print("Training complete!")


if __name__ == '__main__':
    main()
