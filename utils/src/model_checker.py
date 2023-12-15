from colorama import Fore, Style
import colorama
from argparse import ArgumentParser
import os

def model_checker(path_out:str, path_gt:str, verbose:bool = False) -> bool:
    cur_path = os.getcwd().split('src')[0]
    
    file_gt = open(os.path.join(cur_path, path_gt), 'r')
    lines_gt = file_gt.readlines()

    file_out = open(os.path.join(cur_path, path_out), 'r')
    lines_out = file_out.readlines()

    if lines_gt == lines_out:
        if verbose:
            print(Fore.GREEN + Style.BRIGHT + "Correct output")
        return True
    else:
        if verbose:
            print(Fore.RED + "Wrong output")
        return False


def check_all(folder_out:str, folder_gt:str) -> bool:
    cur_path = os.getcwd().split('src')[0]
    print("CHECKING RESULTS")
    print("-"*50)
    wrong_out = False

    count_wrong = 0
    for filename in os.listdir(os.path.join(cur_path, folder_out)):
        path_out = os.path.join(cur_path, folder_out, filename)
        path_gt = os.path.join(cur_path, folder_gt, filename)
        if not model_checker(path_out, path_gt):
            print(Fore.RED + "xxxxx WRONG OUTPUT FOR" + filename + Fore.RESET)
            wrong_out = True
            count_wrong += 1
        else:
            print(Fore.GREEN + "vvvv CORRECT OUTPUT FOR" + filename + Fore.RESET)
    print("Error percentage: " + str(count_wrong/len(os.listdir(os.path.join(cur_path, folder_out)))*100) + "%")
    if not wrong_out:
        print(Fore.GREEN + "----> ALL OUTPUTS ARE CORRECT" + Fore.RESET )
    print("-"*50)
    return not wrong_out


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-o", "--output", dest="output", help="Output folder path", default="results/output")
    parser.add_argument("-g", "--ground_truth", dest="ground_truth", help="Ground truth folder path", default="tests/groundtruths")

    args = parser.parse_args()
    check_all(args.output, args.ground_truth)


