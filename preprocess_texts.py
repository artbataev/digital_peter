import argparse


def convert(text: str) -> str:
    text = " ".join(text.replace(" ", "$"))
    text = text.replace("[", "P").replace("]", "Q")
    return text


def reverse_convert(text: str) -> str:
    text = "".join(text.strip().split(" "))
    text = text.replace("$", " ")
    text = text.replace("P", "[")
    text = text.replace("Q", "]")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outfile")
    parser.add_argument("--in-uttids", action="store_true")
    parser.add_argument("--out-uttids", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()

    with open(args.infile, "r", encoding="utf-8") as f_in, \
            open(args.outfile, "w", encoding="utf-8") as f_out:
        for line in f_in:
            text = line.strip()
            if not text:
                continue
            uttid = ""
            if args.in_uttids:
                try:
                    uttid, text = text.split(maxsplit=1)
                except ValueError:
                    uttid = text
                    text = ""
            text_converted = reverse_convert(text) if args.reverse else convert(text)
            if args.out_uttids:
                print(uttid, text_converted, file=f_out)
            else:
                print(text_converted, file=f_out)
