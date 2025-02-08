#!/usr/bin/env python3
"""
Optimized Collision–List Generator and Search Tool with Logging

Features:
  • Faster collision–list generation with presort and disk saving.
  • Multi–threading / Multi–core processing for collision generation.
  • Multi–key input support (combined via public key addition if on the same curve).
  • Working public key addition between different curves (if they share the same group parameters).
  • Time–to–completion estimates and progress updates.
  • easyCount() search space division with iterative counting and narrowing.
  • Probability / progress updates.
  • Device selection: GPU option available (currently falls back to CPU).

Requirements:
  - Python 3.6+
  - tinyec (pip install tinyec)
  - tqdm (pip install tqdm)

Be aware that many of these features (especially GPU support and the precise tuning of easyCount())
are experimental. Use this as a framework for further tuning.
"""

import os
import sys
import time
import math
import pickle
import concurrent.futures
import multiprocessing
import logging
from os import urandom
from bisect import bisect_left
from tqdm import tqdm
from tinyec.ec import SubGroup, Curve

# Setup logging: change level to DEBUG for more detailed output.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --------------------------- Global Constants ---------------------------
# secp256k1 parameters (default; you can add more curves later)
P = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
N = 115792089237316195423570985008687907852837564279074904382605163141518161494337
A = 0
B = 7
G_POINT = (0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798,
           0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8)
H = 1
CURVE_NAME = 'secp256k1'

# Collision generation parameters (adjust as needed)
MIN_RANGE = 0x4000000000000000000000000000000000  # ~4e36
MAX_RANGE = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  # ~7.9e36
HALF_VAL = 57896044618658097711785492504343953926418782139537452191302581570759080747169
THIRD_VAL = 77194726158210796949047323339125271901891709519383269588403442094345440996225

# Global grid for fast multiplication (built later)
GRID = None
DEVICE = "CPU"  # default; GPU option is available but not fully implemented

# --------------------------- Utility Functions ---------------------------
def check_range(num, min_val, max_val):
    return min_val <= num <= max_val

def check_not_int(value):
    try:
        int(value)
        return False
    except ValueError:
        return True

def check_if_hex(entry):
    try:
        int(entry, 16)
        return True
    except ValueError:
        return False

def combine_public_keys(keys):
    """
    Given a list of public key points (each from tinyec),
    check that they share the same curve parameters and return their sum.
    """
    if not keys:
        raise ValueError("No keys provided")
    base_curve = keys[0].curve
    for k in keys[1:]:
        if k.curve != base_curve:
            logger.error("Public keys are from different curves and cannot be added.")
            sys.exit(1)
    combined = keys[0]
    for k in keys[1:]:
        combined = combined + k
    logger.info("Combined multiple public keys successfully.")
    return combined

def create_key(XX, YY):
    """
    Given X and Y coordinates (as integers), check that they form a valid point
    on the secp256k1 curve and return a public key point (using tinyec).
    """
    if ((XX**3 + 7) % P) != ((YY**2) % P):
        logger.error("Invalid public key coordinates. They are not on the curve.")
        sys.exit(1)
    subgroup = SubGroup(P, (XX, YY), N, H)
    curve = Curve(A, B, subgroup, CURVE_NAME)
    logger.info("Public key created successfully.")
    return curve.g * 1

def binary_search(sorted_array, target):
    """
    Given a sorted list of tuples (each tuple: (collision_x, (offset_from_twos, offset_from_third))),
    find an entry whose first element equals target. Returns the tuple or -1.
    """
    i = bisect_left(sorted_array, (target,))
    if i != len(sorted_array) and sorted_array[i][0] == target:
        return sorted_array[i]
    return -1

# --------------------------- Fast EC “Multiplication” ---------------------------
def build_grid(pub_key):
    """
    Precompute a grid for fast multiplication with the EC generator.
    For each byte position (0–31) compute pub_key * (256^position mod N)
    and then all multiples (0..255) by iterative addition.
    """
    logger.info("Starting to build multiplication grid...")
    grid = []
    for byte_pos in range(32):
        multiplier = pow(256, byte_pos, N)
        base = pub_key * multiplier
        sublist = [0, base]
        current = base
        for _ in range(254):
            current = current + base
            sublist.append(current)
        grid.append(tuple(sublist))
        logger.debug(f"Grid built for byte position {byte_pos}.")
    logger.info("Multiplication grid built successfully.")
    return tuple(grid)

def multiply_num(number):
    """
    Fast “multiplication” of an integer with the EC generator using the precomputed GRID.
    The number is interpreted in 32-byte little-endian form and the corresponding EC points are summed.
    """
    global GRID
    mod_number = number % N
    b_array = mod_number.to_bytes(32, "little")
    total = None
    for i, byte in enumerate(b_array):
        byte_val = byte
        if byte_val:
            if total is None:
                total = GRID[i][byte_val]
            else:
                total = total + GRID[i][byte_val]
    if total is None:
        return "Infinity and Beyond"
    return total

# --------------------------- Collision–List Generation (Parallel) ---------------------------
def process_twos_index(args):
    """
    Process a single two–position (from collision generation).
    Each process computes collision entries for one twos_position.
    Args is a tuple: (index, two_position, AAA, AA, third, N_local)
    Returns a list of tuples: (prefix, (collision_point.x, (offset_from_twos, offset_from_third)))
    """
    index, two_position, AAA, AA, third, N_local = args
    collisions = []
    factor = pow(third, AAA, N_local)
    third_point = two_position * factor
    for third_multiple in range(AAA):
        collision_point = third_point * 3
        try:
            prefix = int(str(collision_point.x)[:6]) - 100000
            if 0 <= prefix < 900000:
                collisions.append((prefix, (collision_point.x, ((index + 1) - AA, (third_multiple + 1) - AAA))))
        except Exception:
            pass
        third_point = collision_point
    return collisions

def sort_keys_parallel(twos_positions, AAA, AA, third, N_local):
    """
    Parallel collision–list generation.
    For each two_position in twos_positions, process collisions in parallel.
    Returns a tuple-of-tuples: one bucket per prefix.
    """
    num_buckets = 900000
    buckets = [[] for _ in range(num_buckets)]
    num_workers = multiprocessing.cpu_count()
    logger.info(f"Using {num_workers} worker processes for collision generation.")
    args_list = []
    for index, two_position in enumerate(twos_positions):
        args_list.append((index, two_position, AAA, AA, third, N_local))
    
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_twos_index, args) for args in args_list]
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Generating Collisions", unit="task"):
            try:
                result = future.result()
                for prefix, data in result:
                    if 0 <= prefix < num_buckets:
                        buckets[prefix].append(data)
            except Exception as e:
                logger.error(f"Error in collision generation: {e}")
    elapsed = time.time() - start_time
    logger.info(f"Collision generation completed in {elapsed:.1f} seconds.")
    sorted_buckets = tuple(tuple(sorted(bucket)) for bucket in buckets)
    return sorted_buckets

# --------------------------- easyCount() Search Implementation ---------------------------
def easy_count_search(collision_list, pub_key, mult_func, N_local):
    """
    Adaptive iterative search (easyCount) which divides the candidate space
    based on the density of collision entries in each bucket.
    """
    logger.info("Starting easyCount() iterative search ...")
    candidate = int.from_bytes(urandom(16), "big") % N_local
    step = 10000  # initial step size; can be tuned
    max_iterations = 10**7
    threshold_high = 5
    threshold_low = 0
    start_time = time.time()
    for i in range(max_iterations):
        try:
            candidate_point = mult_func(candidate)
        except Exception as e:
            logger.error(f"Error computing candidate point: {e}")
            candidate = (candidate + step) % N_local
            continue
        try:
            candidate_x_str = str(candidate_point.x)
        except Exception:
            candidate = (candidate + step) % N_local
            continue
        try:
            bucket_index = int(candidate_x_str[:6]) - 100000
        except Exception:
            candidate = (candidate + step) % N_local
            continue
        if bucket_index < 0 or bucket_index >= len(collision_list):
            candidate = (candidate + step) % N_local
            continue
        bucket = collision_list[bucket_index]
        result = binary_search(bucket, candidate_point.x)
        if result != -1:
            recovered_key = recover_found_key(result, candidate, pub_key, mult_func)
            logger.info("Collision found in easyCount() search!")
            logger.info(f"Found collision x-coordinate: {result[0]}")
            logger.info(f"Recovered Private Key: {recovered_key}")
            with open("foundKeys.txt", "w") as f:
                f.write("-- PrivateKey --\n%s\n-- PublicKey --\n%s\n" % (recovered_key, result[0]))
            return recovered_key
        bucket_len = len(bucket)
        if bucket_len > threshold_high:
            step = max(1, step // 2)
        elif bucket_len == threshold_low:
            step = step * 2
        candidate = (candidate + step) % N_local
        if i % 10000 == 0 and i != 0:
            elapsed = time.time() - start_time
            iterations_per_sec = i / elapsed if elapsed > 0 else 0
            logger.info(f"easyCount iterations: {i}, {iterations_per_sec:.0f} it/sec")
    logger.info("easyCount search did not find a collision in max iterations.")
    return None

# --------------------------- Collision Recovery ---------------------------
def recover_found_key(result, private_key1, input_key, mult_func):
    """
    Given a collision result and a candidate private key, recover the private key.
    """
    twos_offset, threes_offset = result[1]
    factor_two = HALF_VAL if twos_offset >= 0 else 2
    factor_three = THIRD_VAL if threes_offset >= 0 else 3
    exp_two = abs(twos_offset)
    exp_three = abs(threes_offset)
    found_private_key = (private_key1 * (factor_two ** exp_two) * (factor_three ** exp_three)) % N
    corrected_priv_key = mult_func(found_private_key)
    if corrected_priv_key.y != input_key.y:
        found_private_key = N - found_private_key
    return found_private_key

# --------------------------- Collision List Creation ---------------------------
def create_new_collision_list(collision_base_dir):
    """
    Create a new collision list:
      • Ask for one or more public keys (hex X and Y coordinates).
      • (If more than one, combine them via point addition.)
      • Ask for a name and a size parameter.
      • Generate the collision list in parallel and save it (presorted) to disk.
    Returns (collision_list, pub_key, AA)
    """
    multi = input("Do you want to enter multiple public keys? (y/n): ").strip().lower()
    keys = []
    if multi == "y":
        try:
            count = int(input("How many public keys do you want to enter? "))
        except ValueError:
            logger.error("Invalid input.")
            sys.exit(1)
        for i in range(count):
            logger.info(f"Enter public key #{i+1}:")
            XX_hex = input("  X coordinate (hex): ").strip()
            YY_hex = input("  Y coordinate (hex): ").strip()
            if not (check_if_hex(XX_hex) and check_if_hex(YY_hex)):
                logger.error("Invalid hex entry. Exiting.")
                sys.exit(1)
            XX = int(XX_hex, 16)
            YY = int(YY_hex, 16)
            keys.append(create_key(XX, YY))
        pub_key = combine_public_keys(keys)
    else:
        XX_hex = input("Enter your public key X coordinate (hex): ").strip()
        YY_hex = input("Enter your public key Y coordinate (hex): ").strip()
        if not (check_if_hex(XX_hex) and check_if_hex(YY_hex)):
            logger.error("Invalid hex entry. Exiting.")
            sys.exit(1)
        XX = int(XX_hex, 16)
        YY = int(YY_hex, 16)
        pub_key = create_key(XX, YY)
    
    logger.info(f"The public key (in integer form) is: {pub_key}")
    list_name = input("Enter a name for this collision list: ").strip()
    try:
        AA = int(input("Enter a size parameter for the collision list (recommended below 10,000; typical around 1000): "))
    except ValueError:
        logger.error("Invalid integer input. Exiting.")
        sys.exit(1)
    AAA = AA * 2
    total_keys = (AAA + 1) ** 2
    folder_name = f"{list_name}--{total_keys}--{pub_key.x}"
    new_folder = os.path.join(collision_base_dir, folder_name)
    os.makedirs(new_folder, exist_ok=True)
    
    logger.info("Generating collision list ...")
    logger.info(f"Collision list will contain approximately {total_keys} keys.")
    logger.info(f"Optimized for range: {hex(MIN_RANGE)} to {hex(MAX_RANGE)}")
    input("If this is acceptable, press Enter to continue...")
    
    twos_multiplier = (MIN_RANGE + pow(HALF_VAL, AA, N)) % N
    twos_start = pub_key * twos_multiplier
    twos_positions = []
    point = twos_start
    for _ in range(AAA + 1):
        twos_positions.append(point)
        point = point + point
    
    logger.info("Starting parallel collision list generation ...")
    collision_list = sort_keys_parallel(twos_positions, AAA, AA, THIRD_VAL, N)
    collision_file = os.path.join(new_folder, folder_name + ".pkl")
    with open(collision_file, "wb") as f:
        pickle.dump(collision_list, f)
    id_file = os.path.join(new_folder, "idFile.pkl")
    id_data = (pub_key, str((AAA + 1) ** 2))
    with open(id_file, "wb") as f:
        pickle.dump(id_data, f)
    
    logger.info("Collision list generated and saved to disk.")
    return collision_list, pub_key, AA

# --------------------------- Search Routines ---------------------------
def lookup_grid_search(collision_list, pub_key, mult_func):
    """
    Perform a lookup–table grid search for a collision match.
    Returns the recovered private key if found, or None.
    """
    logger.info("Starting lookup grid search ...")
    first_row = [pow(65536, i, N) for i in range(16)]
    first_row = tuple(first_row)
    for easy_grid in tqdm(first_row, ascii=True, ncols=100, unit='Row', desc='Forward Grid'):
        priv_key_val = easy_grid
        try_place = mult_func(priv_key_val)
        easy_grid_val = mult_func(easy_grid)
        easy_key_to_find = try_place.x
        easy_prefix = int(str(easy_key_to_find)[:6])
        for _ in range(65535):
            bucket = collision_list[easy_prefix - 100000]
            result = binary_search(bucket, easy_key_to_find)
            if result != -1:
                recovered_key = recover_found_key(result, priv_key_val, pub_key, mult_func)
                logger.info("Collision match found in lookup grid!")
                logger.info(f"Found collision x-coordinate: {result[0]}")
                logger.info(f"Recovered Private Key: {recovered_key}")
                with open("foundKeys.txt", "w") as f:
                    f.write("-- PrivateKey --\n%s\n-- PublicKey --\n%s\n" % (recovered_key, result[0]))
                return recovered_key
            else:
                try_place = try_place + easy_grid_val
                easy_key_to_find = try_place.x
                easy_prefix = int(str(easy_key_to_find)[:6])
                priv_key_val = (priv_key_val + easy_grid) % N
    return None

def reverse_grid_search(collision_list, pub_key, mult_func):
    """
    Perform reverse grid search from the half–point.
    Returns the recovered private key if found, or None.
    """
    logger.info("Starting reverse lookup grid search ...")
    first_row = [pow(65536, i, N) for i in range(16)]
    first_row = tuple(first_row)
    for reverse_easy in tqdm(first_row, ascii=True, ncols=100, unit='Row', desc='Reverse Grid'):
        priv_key_r = reverse_easy
        reverse_try = mult_func((HALF_VAL - reverse_easy) % N)
        reverse_easy_val = mult_func(reverse_easy)
        reverse_key_to_find = reverse_try.x
        reverse_prefix = int(str(reverse_key_to_find)[:6])
        for _ in range(65535):
            bucket = collision_list[reverse_prefix - 100000]
            result = binary_search(bucket, reverse_key_to_find)
            if result != -1:
                recovered_key_r = recover_found_key(result, priv_key_r, pub_key, mult_func)
                final_key = (HALF_VAL - recovered_key_r) % N
                logger.info("Reverse collision match found!")
                logger.info(f"Found collision x-coordinate: {result[0]}")
                logger.info(f"Recovered Private Key: {final_key}")
                with open("foundKeys.txt", "w") as f:
                    f.write("-- PrivateKey --\n%s\n-- PublicKey --\n%s\n" % (final_key, result[0]))
                return final_key
            else:
                reverse_try = reverse_try - reverse_easy_val
                reverse_key_to_find = reverse_try.x
                reverse_prefix = int(str(reverse_key_to_find)[:6])
                priv_key_r = (priv_key_r - reverse_easy) % N
    return None

def random_search(collision_list, pub_key, mult_func):
    """
    Perform a random search for a collision match.
    Returns the recovered private key if found.
    """
    logger.info("Starting random search for a collision match ...")
    key_found = False
    iterations = 0
    start_round = time.time()
    while not key_found:
        t0 = time.time()
        priv_key_rand = int.from_bytes(urandom(32), "big") % N
        private_key1 = (priv_key_rand - 100000) % N
        keyB = mult_func(private_key1)
        keyG = mult_func(1)
        key_to_find = keyB.x
        prefix = int(str(key_to_find)[:6])
        for _ in range(200000):
            bucket = collision_list[prefix - 100000]
            result = binary_search(bucket, key_to_find)
            iterations += 1
            if result != -1:
                recovered_key_random = recover_found_key(result, private_key1, pub_key, mult_func)
                logger.info("Random search collision found!")
                logger.info(f"Found collision x-coordinate: {result[0]}")
                logger.info(f"Recovered Private Key: {recovered_key_random}")
                with open("foundKeys.txt", "w") as f:
                    f.write("-- PrivateKey --\n%s\n-- PublicKey --\n%s\n" % (recovered_key_random, result[0]))
                return recovered_key_random
            else:
                keyB = keyB + keyG
                key_to_find = keyB.x
                prefix = int(str(key_to_find)[:6])
                private_key1 = (private_key1 + 1) % N
        elapsed = time.time() - t0
        iter_rate = 200000 / elapsed if elapsed > 0 else 0
        logger.info(f"Random search: {iter_rate:.0f} it/sec (total iterations: {iterations})")
    return None

# --------------------------- Main Routine ---------------------------
def main():
    global GRID, DEVICE

    logger.info("Starting Collision Finder Script")
    print("\n" * 2)
    print("\033[0;32m _               _        _       _     _\033[00m")
    print("\033[0;32m| |__   __ _ ___| |__    / \\   __| | __| | ___ _ __\033[00m")
    print("\033[0;32m| '_ \\ / _` / __| '_ \\  / _ \\ / _` |/ _` |/ _ \\ '__|\033[00m")
    print("\033[0;32m| | | | (_| \\__ \\ | | |/ ___ \\ (_| | (_| |  __/ |\033[00m")
    print("\033[0;32m|_| |_|\\__,_|___/_| |_/_/   \\_\\__,_|\\__,_|\\___|_|\033[00m")
    print("\n" * 2)
    
    device_choice = input("Select computation device ([CPU] or GPU): ").strip().upper()
    if device_choice == "GPU":
        logger.warning("GPU support is not yet fully implemented; falling back to CPU.")
        DEVICE = "CPU"
    else:
        DEVICE = "CPU"
    
    base_dir = os.getcwd()
    collision_dir = os.path.join(base_dir, "collisionLists")
    if not os.path.exists(collision_dir):
        os.mkdir(collision_dir)
    
    subdirs = [d for d in os.listdir(collision_dir) if os.path.isdir(os.path.join(collision_dir, d))]
    collision_list = None
    pub_key_result = None
    AA = None
    if subdirs:
        logger.info("Existing collision lists found.")
        print("(C) -- Create a New Collision List")
        valid_dirs = []
        for idx, folder in enumerate(subdirs):
            if folder == "idFile.pkl":
                continue
            valid_dirs.append(folder)
            print(f"({idx + 1}) {folder}")
        choice = input("Enter the number of the collision list to load or 'C' to create a new one: ").strip()
        if choice.upper() != "C":
            if check_not_int(choice):
                logger.error("Invalid entry. Exiting.")
                sys.exit(1)
            choice = int(choice)
            if not check_range(choice, 1, len(valid_dirs)):
                logger.error("Invalid range. Exiting.")
                sys.exit(1)
            selected = valid_dirs[choice - 1]
            list_dir = os.path.join(collision_dir, selected)
            collision_file = os.path.join(list_dir, selected + ".pkl")
            logger.info("Loading selected collision list (this may take a moment)...")
            with open(collision_file, "rb") as f:
                collision_list = pickle.load(f)
            id_file = os.path.join(list_dir, "idFile.pkl")
            with open(id_file, "rb") as f:
                id_data = pickle.load(f)
                pub_key_result = id_data[0]
                AA = int(id_data[1])
        else:
            collision_list, pub_key_result, AA = create_new_collision_list(collision_dir)
    else:
        collision_list, pub_key_result, AA = create_new_collision_list(collision_dir)
    
    logger.info("Building multiplication grid for fast EC operations...")
    GRID = build_grid(pub_key_result)
    os.chdir(base_dir)
    
    skip_lookup = input("Would you like to skip the lookup grid search? (y to skip, n to run): ").strip().lower()
    recovered = None
    if skip_lookup != "y":
        recovered = lookup_grid_search(collision_list, pub_key_result, multiply_num)
        if recovered is not None:
            sys.exit(0)
        recovered = reverse_grid_search(collision_list, pub_key_result, multiply_num)
        if recovered is not None:
            sys.exit(0)
    else:
        logger.info("Skipping lookup grid search.")
    
    if input("Would you like to run easyCount() search (adaptive iterative narrowing)? (y/n): ").strip().lower() == "y":
        recovered = easy_count_search(collision_list, pub_key_result, multiply_num, N)
        if recovered is not None:
            sys.exit(0)
    
    recovered = random_search(collision_list, pub_key_result, multiply_num)
    if recovered is None:
        logger.info("No collision found.")
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
