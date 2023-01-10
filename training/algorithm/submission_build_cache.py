from misc_utilities.build_cache import main as build_cache

build_cache("/input", "/scratch/auto_cache", num_workers=20, outer_size=256, inner_size=224)
