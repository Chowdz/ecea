import argparse


class Options:
    def __init__(self):
        self.IMG_SIZE: int = 256
        self.IN_C: int = 4
        self.OUT_C: int = 3
        self.PATCH_SIZE: int = 4
        self.EMBED_DIM: int = 64
        self.DEPTH: list = [1, 2, 3, 4]
        self.NUM_HEADS: list = [1, 2, 4, 8]

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--INPUT_IMG_PATH', type=str, help='Input image path')
        parser.add_argument('--OUTPUT_IMG_PATH', type=str, help='Output image path')
        parser.add_argument('--VIS_PARAM', type=dict, help='Output image path')
        args = parser.parse_args()
        return args
