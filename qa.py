import argparse

from ask import ask_question

parser = argparse.ArgumentParser(
    description='Ask a question to the Lushu Book.')
parser.add_argument('question', type=str,
                    help='The question to ask the Lushu Book')
args = parser.parse_args()

result = ask_question(args.question)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
