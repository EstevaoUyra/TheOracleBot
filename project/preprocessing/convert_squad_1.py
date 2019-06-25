import argparse
import json
import logging
import sys


class ParserWithUsage(argparse.ArgumentParser):
    """ A custom parser that writes error messages followed by command line usage documentation."""

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def get_sentence_id_for_span(sentence_spans, span_start):
    for index, (_, end) in enumerate(sentence_spans):
        if span_start <= end:
            return index


def tokenize_and_split_sentences(text: str):
    import nltk
    sentence_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    sentence_spans = list(sentence_tokenizer.span_tokenize(text))
    sentences = sentence_tokenizer.tokenize(text)
    value_to_return = {"sentence_spans": sentence_spans, "sentences": sentences}
    return value_to_return


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')
    parser = ParserWithUsage()
    parser.description = "Reads the SQuAD 1 and converts it to JSON file containing question, sentence, is_answer"
    parser.add_argument("--input", help="Input SQuAD JSON file", required=True)
    parser.add_argument("--output", help="Output file", required=True)

    args = parser.parse_args()
    logging.info("STARTED")

    output_data = []
    with open(args.input, "r") as original_file:
        original_data = json.load(original_file)["data"]
        for current_document in original_data:
            for current_paragraph in current_document["paragraphs"]:
                processed_context = tokenize_and_split_sentences(current_paragraph["context"])
                for current_question in current_paragraph["qas"]:
                    question_text = current_question["question"]
                    sentences_containing_answer = set(
                        [get_sentence_id_for_span(processed_context["sentence_spans"], a["answer_start"]) for a in
                         current_question["answers"]])
                    for index, s in enumerate(processed_context["sentences"]):
                        example = {"question": question_text, "sentence": s,
                                   "label": int(index in sentences_containing_answer)}
                        output_data.append(example)
                        if len(output_data) % 1000 == 0:
                            logging.info("Processed {} documents".format(len(output_data)))
    logging.info("Writing to file {}".format(args.output))
    with open(args.output, "w") as output_file:
        json.dump(output_data, output_file, indent=4)
    logging.info("DONE")


if __name__ == "__main__":
    main()
