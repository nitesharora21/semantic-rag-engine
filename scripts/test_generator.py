from rag_engine.generator import OllamaGenerator

def main() -> None:
    generator = OllamaGenerator(model_name="gemma3")
    answer = generator.generate(
            "Explain retrieval-augmented generatio in two sentences"
    )
    print(answer)

if __name__ == "__main__":
    main()
