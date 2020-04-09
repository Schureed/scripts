import argparse
import heapq
import tqdm
import multiprocessing as mp
import torch

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--span', type=int, default=3)
    parser.add_argument('--dropout-p', type=float, default=0.1)
    parser.add_argument('--blank-p', type=float, default=0.1)
    return parser


def count_line(filename):
    import subprocess
    return int(subprocess.check_output("wc -l %s" % filename, shell=True).decode('utf8').split()[0])


def add_noise(args):
    line, args = args
    def word_shuffle(line, span):
        permutation = torch.arange(len(line)).float().add(torch.rand(len(line)) * span).sort()[1]
        return [line[i] for i in permutation]

    def word_dropout(line, p):
        keep = torch.rand(len(line))
        res = [si for i, si in enumerate(line) if keep[i] > p]
        if len(res) == 0:
            return line[len(line)//2]
        return res

    def word_blank(line, p):
        keep = torch.rand(len(line))
        return [si if keep[i] > p else '<unk>' for i, si in enumerate(line)]

    tokens = line.split()
    tokens = word_shuffle(tokens, args.span)
    tokens = word_dropout(tokens, args.dropout_p)
    tokens = word_blank(tokens, args.blank_p)
    return " ".join(tokens)

if __name__ == "__main__":
  args = build_parser().parse_args()
  chunk_size = 8
  lines = count_line(args.input)
  cpus = mp.cpu_count() * chunk_size

  with open(args.input) as read, \
        open(args.output, 'w') as write, \
        mp.Pool() as pool:
    bar = tqdm.tqdm(total=lines)

    for i in range(0, lines - cpus, cpus):
        buf = [(read.readline(), args) for _ in range(cpus)]
        res = pool.map(add_noise, buf, chunk_size)

        for line in res:
            write.write(line)
            write.write('\n')
        if i % (cpus * 1000) == 0:
            write.flush()
        bar.update(cpus)
    for i in range(lines // cpus * cpus, lines):
        write.write(add_noise((read.readline(), args)))
        write.write('\n')
        bar.update()



