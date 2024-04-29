import json
import os
from tqdm import tqdm

def load_questions(filename):
    """ Load questions from a JSON file. """
    with open(filename, 'r', encoding='utf-8') as file:
        questions = json.load(file)
    return questions

def save_answers(answers, filename):
    """ Save answers to a JSON file. """
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(answers, file, ensure_ascii=False, indent=2)

def main(corpus_files, question_file, answer_file):
    # Initialize the ChatPDF system
    sim_model = BM25Similarity()  # or other similarity models
    model = ChatPDF(
        similarity_model=sim_model,
        generate_model_type='auto',
        generate_model_name_or_path='01-ai/Yi-6B-Chat',
        corpus_files=corpus_files,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # Add multiple PDFs to the corpus
    model.add_corpus(corpus_files)

    # Load questions
    questions = load_questions(question_file)

    # Process each question and collect answers
    answers = []
    for question in tqdm(questions, desc="Processing questions"):
        response, _ = model.predict(question['question'])
        answer = {
            'id': question['id'],
            'question': question['question'],
            'answer': response
        }
        answers.append(answer)

    # Save answers to a JSON file
    save_answers(answers, answer_file)

if __name__ == "__main__":
    # Define paths to your files
    corpus_directory = '/path/to/your/pdf/documents/'
    corpus_files = [os.path.join(corpus_directory, f) for f in os.listdir(corpus_directory) if f.endswith('.pdf')]
    question_file = 'path/to/question_example.json'
    answer_file = 'path/to/answer_example.json'

    # Run the main function
    main(corpus_files, question_file, answer_file)
