from pathlib import Path


def clean_text(text, chars_to_remove):
    replacements = [
        ("x", "х"),
        ("ѳ", "ф"),
        ("ѡ", "о"),
        ("і", "i"),
        ("ѯ", "кс"),
    ]
    strange_chars = {"ѱ", "є", "v"}
    for char_in, char_out in replacements:
        text = text.replace(char_in, char_out)
    for char in chars_to_remove:
        text = text.replace(char, " ")
    text = " ".join(word for word in text.split() if len(set(word).intersection(strange_chars)) == 0)
    return text


if __name__ == "__main__":
    texts = set()
    with open("data/grameval_17_century.txt", "r", encoding="utf-8") as f:
        for line in f:
            texts.add(" ".join(line.strip().split()))
    chars = {" "}
    with open("data/chars_new.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chars.add(line.strip())
    char_geval = set("".join(texts))
    chars_to_remove = char_geval - chars - {"ѱ", "є", "v", "x", "ѳ", "ѡ", "і", "ѯ"}
    print(char_geval - chars)
    print(chars - char_geval)
    texts = set(clean_text(text, chars_to_remove) for text in texts) - {""}
    char_geval = set("".join(texts))
    print(char_geval - chars)
    with open("data/grameval_17_century_cleaned.txt", "w", encoding="utf-8") as f_texts, \
            open("data/grameval_17_century_cleaned_for_lm.txt", "w", encoding="utf-8") as f_texts_lm:
        for text in texts:
            print(text, file=f_texts)
            print(" ".join(text.replace(" ", "$")), file=f_texts_lm)
