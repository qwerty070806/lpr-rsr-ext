import os
import argparse
from pathlib import Path
from __syslog__ import EventlogHandler


# ------------------------------------------------------------
# TRAIN / LOAD / EVAL ARGUMENTS
# ------------------------------------------------------------

@EventlogHandler
def args_m01():

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-t", "--samples",
        type=Path,
        required=True,
        help="Path to dataset root directory"
    )

    ap.add_argument(
        "-s", "--save",
        type=Path,
        required=True,
        help="Checkpoint save directory"
    )

    ap.add_argument(
        "-b", "--batch",
        type=int,
        required=True,
        help="Batch size"
    )

    ap.add_argument(
        "-m", "--mode",
        type=int,
        required=True,
        choices=[0, 1, 2],
        help="Reset=0 | Load=1 | Evaluate=2"
    )

    ap.add_argument(
        "--model",
        type=Path,
        required=False,
        default=Path(""),
        help="Checkpoint path for resume / eval"
    )

    try:
        args = ap.parse_args()
    except:
        ap.print_help()
        raise Exception("Missing Arguments")

    # ----------------------------
    # PATH CHECKS
    # ----------------------------

    if not args.samples.exists():
        raise argparse.ArgumentTypeError(
            "'--samples' path does not exist."
        )

    if args.mode == 0:

        if not args.save.exists():
            print(f"Creating save directory: {args.save}")
            args.save.mkdir(parents=True, exist_ok=True)

        args.model = None

    elif args.mode in [1, 2]:

        if not args.model.is_file():
            raise argparse.ArgumentTypeError(
                "'--model' must be a valid .pt checkpoint file"
            )

        if not args.save.exists():
            raise argparse.ArgumentTypeError(
                "'--save' directory must exist."
            )

    return args


# ------------------------------------------------------------
# TEST / EVAL ONLY
# ------------------------------------------------------------

@EventlogHandler
def args_m2():

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-t", "--samples",
        type=Path,
        required=True,
        help="Path to dataset root directory"
    )

    ap.add_argument(
        "-s", "--save",
        type=Path,
        required=True,
        help="Evaluation output directory"
    )

    ap.add_argument(
        "-b", "--batch",
        type=int,
        required=True,
        help="Batch size"
    )

    ap.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Checkpoint (.pt) to evaluate"
    )

    try:
        args = ap.parse_args()
    except:
        ap.print_help()
        raise Exception("Missing Arguments")

    # ----------------------------
    # PATH CHECKS
    # ----------------------------

    if not args.samples.exists():
        raise argparse.ArgumentTypeError(
            "'--samples' path does not exist."
        )

    if not args.save.exists():
        print(f"Creating evaluation directory: {args.save}")
        args.save.mkdir(parents=True, exist_ok=True)

    if not args.model.is_file():
        raise argparse.ArgumentTypeError(
            "'--model' must be a valid .pt checkpoint file."
        )

    return args


if __name__ == "__main__":
    args_m01()
