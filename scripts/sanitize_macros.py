from os import walk
from os.path import join


def read_macro_blocks(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        macro_block = False
        macro_blocks = list()
        buffer = list()
        for nr, line in enumerate(file):
            line = line.replace('\n', '')
            if line.endswith("\\"):
                buffer.append((nr, line))
                macro_block = True
                continue

            if macro_block is True:
                buffer.append((nr, line))
                macro_block = False
                macro_blocks.append(buffer)
                buffer = list()
    return macro_blocks


def print_macro_blocks(macro_blocks):
    for block in macro_blocks:
        print("=" * 150)
        for nr, line in block:
            print(f"{nr:>4d}: {line}")
        print("=" * 150)


def process_macro_block(macro_block):
    def rreplace(s, old, new, occurrence=-1):
        if occurrence < 0:
            occurrence = s.count(old) + 1 + occurrence
        li = s.rsplit(old, occurrence)
        return new.join(li)

    macro_block = [(nr, rreplace(line, '\\', '').rstrip().replace('\t', 4 * ' ')) for nr, line in macro_block]
    max_line = max(len(line) for _, line in macro_block)
    macro_block = [(nr, (line + ' ' * (max_line * 2))[:max_line + 3]) for nr, line in macro_block]
    result = [(nr, line + '\\') for nr, line in macro_block[:-1]] + [(macro_block[-1][0], macro_block[-1][1].rstrip())]
    return result


def process_macro_blocks(macro_blocks):
    return [process_macro_block(macro_block) for macro_block in macro_blocks]


def save_macro_blocks(in_file_name, out_file_name, flattened_blocks):
    i = 0
    with open(in_file_name, 'r', encoding='utf-8') as in_file:
        in_lines = in_file.readlines()

    with open(out_file_name, 'w', encoding='utf-8') as out_file:
        for nr, line in enumerate(in_lines):
            line = line.replace('\n', '')
            if i < len(flattened_blocks):
                block_nr, block_line = flattened_blocks[i]
            else:
                block_nr, block_line = -1, ''

            if nr == block_nr:
                print(block_line, file=out_file)
                i += 1
            else:
                print(line, file=out_file)


def sanitize_file(file_name):
    macro_blocks = read_macro_blocks(file_name)
    if len(macro_blocks) == 0:
        return False
    new_macro_blocks = process_macro_blocks(macro_blocks)
    if macro_blocks == new_macro_blocks:
        return False

    flattened_blocks = [(nr, line) for macro_block in new_macro_blocks for nr, line in macro_block]
    save_macro_blocks(file_name, file_name, flattened_blocks)
    return True


def main():
    print("Start macro block formatting...")
    directories = ("./benchmarks", "./include", "./src", "./tests")

    scanned, sanitized = 0, 0

    for directory in directories:
        for root, directories, files in walk(directory):
            for file in files:
                if file.endswith(("hxx", "hpp", "cxx", "cpp")):
                    file_name = join(root, file)
                    scanned += 1
                    if sanitize_file(file_name):
                        sanitized += 1
                        print(f"    Sanitized macro blocks in '{file_name}'")

    print("Finished")
    print(f"Sanitized C++ files: {sanitized} / {scanned} ({100 * sanitized / scanned:>5.2f}%)")


if __name__ == "__main__":
    main()
