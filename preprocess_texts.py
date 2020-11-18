import argparse


def convert(text: str) -> str:
    text = " ".join(text.replace(" ", "$"))
    text = text.replace("[", "P").replace("]", "Q")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outfile")
    parser.add_argument("--strip-uttids", action="store_true")
    args = parser.parse_args()
    with open(args.infile, "r", encoding="utf-8") as f_in, \
            open(args.outfile, "w", encoding="utf-8") as f_out:
        for line in f_in:
            text = line.strip()
            if args.strip_uttids:
                text = text.split(maxsplit=1)[1]
            print(convert(text), file=f_out)
