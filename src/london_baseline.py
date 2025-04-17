# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0
    argp = argparse.ArgumentParser()
    argp.add_argument('--eval_corpus_path')
    args = argp.parse_args()
    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###

    with open(args.eval_corpus_path) as file:
        lines = file.readlines()
        predictions = ["London"] * len(lines)
    total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    ### END YOUR CODE ###
    accuracy = correct / total
    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
    print(f"Baseline accuracy: {accuracy}")
