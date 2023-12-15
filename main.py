import argparse
from textual_information_processor import TextualInformationProcessor

def main():
    parser = argparse.ArgumentParser(description="Textual Information Processor")
    parser.add_argument("input_file", help="Input CSV file path")
    parser.add_argument("output_file", help="Output CSV file path")
    parser.add_argument("--chunksize", type=int, default=100, help="Chunk size (default: 100)")
    parser.add_argument("--text_column", help="Name of the text column in the CSV")
    parser.add_argument("--guidance_column", help="Name of the text column in the CSV")

    args = parser.parse_args()

    processor = TextualInformationProcessor(args.input_file, args.output_file)

    if args.text_column:
        processor.set_text_column(args.text_column)

    if args.guidance_column:
        processor.set_guidance_column(args.guidance_column)


    processor.process_data(chunksize=args.chunksize)

if __name__ == '__main__':
    main()
