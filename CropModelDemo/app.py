"""
app.py
------
Run this to chat with the crop assistant on the command line.

    python app.py

Type a question about Indian crop production, yield, or shortages
(2010-2017 data). Type 'quit' or 'exit' to stop.

Example questions:
    What was rice production in Punjab in 2015?
    Did we have a pulse shortage in 2015?
    Predict wheat production in Haryana for 2018.
    How much cotton did Gujarat produce?
"""

from nlp_interface import CropAssistant


BANNER = """
=========================================================
  Indian Crop Data Assistant  (demo model, data 2010-2017)
=========================================================
Ask a question, e.g.:
  - What was rice production in Punjab in 2015?
  - Did we have a pulse shortage in 2015?
  - Predict wheat production in Haryana for 2018.

Type 'quit' to exit.
---------------------------------------------------------
"""


def main():
    assistant = CropAssistant()
    print(BANNER)
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        answer = assistant.answer(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
