from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def load_model(model_path):
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

def main():
    # Yeni modeli yükleyin
    qa_pipeline = load_model("ytu-ce-cosmos/turkish-large-bert-cased")

    print("Merhaba, Ben FinBot!")
    print("Sorularınızı sorun veya çıkmak için 'q' yazın.")

    while True:
        question = input("\nSorunuz: ")
        if question.lower() == 'q':
            break

        # Geliştirilmiş bağlam
        context = """
        Türkiye ekonomisi, tarım, sanayi ve hizmet sektörlerinden oluşmaktadır. 
        Enflasyon, işsizlik oranları ve döviz kurları önemli ekonomik göstergelerdir.
        Türkiye'nin en büyük ticaret ortakları Almanya, Çin ve İtalya'dır.
        2021 yılında Türkiye'nin Gayri Safi Yurtiçi Hasılası 814 milyar dolar olarak kaydedilmiştir.
        Ülke, turizm, tekstil ve otomotiv sektörlerinde önemli bir yere sahiptir.
        """

        result = qa_pipeline(question=question, context=context)
        print(f"\nCevap: {result['answer']}")
        print(f"Güven skoru: {result['score']:.2f}")

    print("Ekonomi Botu kapatılıyor. İyi günler!")


if __name__ == "__main__":
    main()
