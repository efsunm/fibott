from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    # CSV dosyasını yükleyin
    df = pd.read_csv(file_path)
    
    # Başlangıç ve bitiş pozisyonlarını ekleyin
    df[['start_positions', 'end_positions']] = df.apply(get_positions, axis=1)
    
    return df

def get_positions(row):
    # Cevap pozisyonlarını hesaplayın
    start = row['soru'].find(row['cevap'])
    end = start + len(row['cevap'])
    return pd.Series([start, end])

def load_glossary(file_path):
    # Ekonomi sözlüğünü yükleyin
    glossary_df = pd.read_csv(file_path)
    glossary_df.columns = ['term', 'description']  # Sütun adlarını ayarla
    return glossary_df

def prepare_data(df):
    # Veriyi eğitim ve doğrulama setlerine ayır
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, val_df

def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples["soru"],
        examples["cevap"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    if "start_positions" in examples and "end_positions" in examples:
        tokenized["start_positions"] = examples["start_positions"]
        tokenized["end_positions"] = examples["end_positions"]

    return tokenized

def main():
    # Veri setini yükle
    df = load_data("/Applications/FinBot/turkiye_ekonomi_bot/data/ekonomi_qa.csv")
    train_df, val_df = prepare_data(df)

    # Ekonomi sözlüğünü yükle
    glossary_df = load_glossary("/Applications/FinBot/turkiye_ekonomi_bot/data/ekonomi_sozluk.csv")

    # Model ve tokenizer'ı yükle
    model_name = "ytu-ce-cosmos/turkish-large-bert-cased"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Veri setlerini oluştur
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Veriyi tokenize et
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Eğitim argümanlarını tanımla
    training_args = TrainingArguments(
    output_dir="./models",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=8,      #daha fazla eğitiyon işte arttırırsan 
    per_device_train_batch_size=5,  #daha fazla veri 8 ya da 16 dene 
    per_device_eval_batch_size=5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    learning_rate=5e-5,
)


    # Trainer'ı oluştur ve eğitimi başlat
    trainer = Trainer(
        model=model,
        args=training_args    ,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Eğitimi başlat
    trainer.train()

    # Eğitilmiş modeli kaydet
    model.save_pretrained("./models/turkiye_ekonomi_model")
    tokenizer.save_pretrained("./models/turkiye_ekonomi_model")

if __name__ == "__main__":
    main()
