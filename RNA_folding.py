# Copyright 2021 D-Wave Systems
# Based on the paper 'RNA folding using quantum computers'
# Fox DM, MacDermaid CM, Schreij AM, Zwierzyna M, Walker RC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import dirname, join
from collections import defaultdict
from itertools import product, combinations

import tempfile
import os
import click
import matplotlib
import numpy as np
import networkx as nx
import dimod
import math
from typing import Optional
from dwave.system import LeapHybridCQMSampler
try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

"""
Possible reduction strategies for RNA folding:
1. Collapse idle sequence of nucleotides, considering them as a single base with higher energy
2. Leverage Repetition or symmetry of nucleotides
3. Dynamic programming, Cut and find optimal sting configuration for each.
4. Pruning of base pairs with high energy, assuming we do not change the solution too much
5. Merge maximal contiguous stems to a single super node, effectively reducing the problem size.
"""

def reduce_palindrome_symmetry(bond_matrix, rna_sequence, symmetry_factor=2, min_len=10):
    """
    Reduce problem by merging palindrome halves into super-nodes.
    
    Args:
        bond_matrix (:class: `numpy.ndarray`):
            Original bond matrix.
        rna_sequence (str):
            The RNA sequence string.
        symmetry_factor (int):
            Factor to multiply bond strengths for symmetric regions.
        min_len (int):
            Minimum length for palindrome to be considered for reduction (default: 10).
            
    Returns:
        dict: Results dictionary with 'reduced_matrix', 'position_mapping', 'reverse_mapping', 
              'modified_sequence', 'symmetry', and 'applied' fields, or None if no valid palindromes found
    """
    # Find palindromes in the sequence
    symmetries = find_sequence_symmetries(rna_sequence, min_len)
    palindromes = symmetries['palindromes']
    
    if not palindromes:
        print(f"  No palindromes found with min_len={min_len}")
        return None
    
    print(f"  Found {len(palindromes)} palindrome(s) with min_len={min_len}: {palindromes}")
    
    # Use the longest palindrome for reduction
    palindrome = max(palindromes, key=lambda x: x[2])  # Sort by length (index 2)
    start, end, length = palindrome
    n = bond_matrix.shape[0]
    
    # Check if palindrome meets minimum length requirement
    if length < min_len:
        return None
    
    # Validate palindrome coordinates
    if start < 0 or end >= n or start > end:
        return None
    
    # Calculate mid-point of palindrome
    mid_point = start + length // 2
    
    # Define the two halves of the palindrome
    first_half = list(range(start, mid_point))
    second_half = list(range(mid_point + 1, end + 1))
    
    # Handle center position (for both odd and even length palindromes)
    center_position = mid_point
    # For odd-length palindromes, center is shared; for even-length, it's the boundary
    if length % 2 == 1:
        # Remove center from both halves for odd-length palindromes
        if center_position in first_half:
            first_half.remove(center_position)
        if center_position in second_half:
            second_half.remove(center_position)
    else:
        # For even-length palindromes, center position is not in either half
        # but still needs to be mapped
        pass
    
    # Create position mapping
    position_mapping = {}
    reverse_mapping = {}
    new_pos = 0
    
    # Map positions outside palindrome normally
    for i in range(n):
        if i < start or i > end:
            position_mapping[i] = new_pos
            reverse_mapping[new_pos] = [i]
            new_pos += 1
    
    # Map first half of palindrome to a super-node
    if first_half:
        for pos in first_half:
            position_mapping[pos] = new_pos
        reverse_mapping[new_pos] = first_half
        new_pos += 1
    
    # Map center position (for both odd and even length palindromes)
    position_mapping[center_position] = new_pos
    reverse_mapping[new_pos] = [center_position]
    new_pos += 1
    
    # Map second half of palindrome to a super-node
    if second_half:
        for pos in second_half:
            position_mapping[pos] = new_pos
        reverse_mapping[new_pos] = second_half
        new_pos += 1
    
    # Create reduced matrix
    reduced_size = new_pos
    reduced_matrix = np.zeros((reduced_size, reduced_size), dtype=int)
    
    # Fill reduced matrix with bond strengths
    for i_orig in range(n):
        for j_orig in range(i_orig + 1, n):
            if bond_matrix[i_orig, j_orig] > 0:
                new_i, new_j = position_mapping[i_orig], position_mapping[j_orig]
                if new_i != new_j:
                    # Check if both positions are in palindrome halves
                    i_in_first_half = i_orig in first_half
                    i_in_second_half = i_orig in second_half
                    j_in_first_half = j_orig in first_half
                    j_in_second_half = j_orig in second_half
                    
                    # Enhanced bond strength for symmetric regions
                    bond_strength = bond_matrix[i_orig, j_orig]
                    
                    # If both positions are in palindrome halves, enhance bond strength
                    if ((i_in_first_half and j_in_second_half) or 
                        (i_in_second_half and j_in_first_half)):
                        bond_strength *= symmetry_factor
                    
                    reduced_matrix[min(new_i, new_j), max(new_i, new_j)] += bond_strength
    
    # Create modified sequence
    modified_sequence = []
    i = 0
    while i < len(rna_sequence):
        if start <= i <= end:
            # We're inside the palindrome
            if i < mid_point:
                # First half - add 'P' for first half of palindrome
                modified_sequence.append('P')
                i = mid_point
            elif i == mid_point and length % 2 == 1:
                # Center position for odd-length palindrome
                modified_sequence.append(rna_sequence[i])
                i += 1
            elif i > mid_point:
                # Second half - add 'Q' for second half of palindrome
                modified_sequence.append('Q')
                i = end + 1
            else:
                i += 1
        else:
            # Outside palindrome - add normally
            modified_sequence.append(rna_sequence[i])
            i += 1
    
    modified_sequence = ''.join(modified_sequence)
    
    return {
        'reduced_matrix': reduced_matrix,
        'position_mapping': position_mapping,
        'reverse_mapping': reverse_mapping,
        'modified_sequence': modified_sequence,
        'symmetry': palindrome,
        'applied': True
    }

def reduce_repeat_symmetry(bond_matrix, rna_sequence, repeat_factor=1.5, min_len=4):
    """
    Reduce problem by compressing repeated patterns into super-nodes.
    
    Args:
        bond_matrix (:class: `numpy.ndarray`):
            Original bond matrix.
        rna_sequence (str):
            The RNA sequence string.
        repeat_factor (float):
            Factor to multiply bond strengths for connections between repeats.
        min_len (int):
            Minimum length for repeat to be considered for reduction (default: 4).
            
    Returns:
        dict: Results dictionary with 'reduced_matrix', 'position_mapping', 'reverse_mapping', 
              'modified_sequence', 'symmetry', and 'applied' fields, or None if no valid repeats found
    """
    # Find repeats in the sequence
    symmetries = find_sequence_symmetries(rna_sequence, min_len)
    repeats = symmetries['repeats']
    
    if not repeats:
        print(f"  No repeats found with min_len={min_len}")
        return None
    
    print(f"  Found {len(repeats)} repeat(s) with min_len={min_len}: {repeats}")
    
    # Use the longest repeat for reduction
    repeat = max(repeats, key=lambda x: x[4])  # Sort by length (index 4)
    start1, end1, start2, end2, length = repeat
    n = bond_matrix.shape[0]
    
    # Check if repeat meets minimum length requirement
    if length < min_len:
        return None
    
    # Define the repeated regions
    first_repeat = list(range(start1, end1 + 1))
    second_repeat = list(range(start2, end2 + 1))
    
    # Create position mapping
    position_mapping = {}
    reverse_mapping = {}
    new_pos = 0
    
    # Map positions before first repeat normally
    for i in range(start1):
        position_mapping[i] = new_pos
        reverse_mapping[new_pos] = [i]
        new_pos += 1
    
    # Map first repeat to a super-node
    for pos in first_repeat:
        position_mapping[pos] = new_pos
    reverse_mapping[new_pos] = first_repeat
    new_pos += 1
    
    # Map positions between repeats normally
    for i in range(end1 + 1, start2):
        position_mapping[i] = new_pos
        reverse_mapping[new_pos] = [i]
        new_pos += 1
    
    # Map second repeat to a super-node
    for pos in second_repeat:
        position_mapping[pos] = new_pos
    reverse_mapping[new_pos] = second_repeat
    new_pos += 1
    
    # Map positions after second repeat normally
    for i in range(end2 + 1, n):
        position_mapping[i] = new_pos
        reverse_mapping[new_pos] = [i]
        new_pos += 1
    
    # Create reduced matrix
    reduced_size = new_pos
    reduced_matrix = np.zeros((reduced_size, reduced_size), dtype=int)
    
    # Fill reduced matrix with bond strengths
    for i_orig in range(n):
        for j_orig in range(i_orig + 1, n):
            if bond_matrix[i_orig, j_orig] > 0:
                new_i, new_j = position_mapping[i_orig], position_mapping[j_orig]
                if new_i != new_j:
                    # Check if both positions are in repeat regions
                    i_in_first_repeat = i_orig in first_repeat
                    i_in_second_repeat = i_orig in second_repeat
                    j_in_first_repeat = j_orig in first_repeat
                    j_in_second_repeat = j_orig in second_repeat
                    
                    # Enhanced bond strength for connections between repeats
                    bond_strength = bond_matrix[i_orig, j_orig]
                    
                    # If both positions are in different repeat regions, enhance bond strength
                    if ((i_in_first_repeat and j_in_second_repeat) or 
                        (i_in_second_repeat and j_in_first_repeat)):
                        bond_strength = int(bond_strength * repeat_factor)
                    
                    reduced_matrix[min(new_i, new_j), max(new_i, new_j)] += bond_strength
    
    # Create modified sequence
    modified_sequence = []
    i = 0
    while i < len(rna_sequence):
        if start1 <= i <= end1:
            # First repeat - add 'R' for repeat
            modified_sequence.append('R')
            i = end1 + 1
        elif start2 <= i <= end2:
            # Second repeat - add 'R' for repeat
            modified_sequence.append('R')
            i = end2 + 1
        else:
            # Outside repeats - add normally
            modified_sequence.append(rna_sequence[i])
            i += 1
    
    modified_sequence = ''.join(modified_sequence)
    
    return {
        'reduced_matrix': reduced_matrix,
        'position_mapping': position_mapping,
        'reverse_mapping': reverse_mapping,
        'modified_sequence': modified_sequence,
        'symmetry': repeat,
        'applied': True
    }

def reduce_mirror_symmetry(bond_matrix, rna_sequence, complementarity_factor=2, min_len=3):
    """
    Reduce problem by merging complementary sequences into binding pair super-nodes.
    
    Args:
        bond_matrix (:class: `numpy.ndarray`):
            Original bond matrix.
        rna_sequence (str):
            The RNA sequence string.
        complementarity_factor (int):
            Factor to multiply bond strengths for complementary regions.
        min_len (int):
            Minimum length for mirror symmetry to be considered for reduction (default: 3).
            
    Returns:
        dict: Results dictionary with 'reduced_matrix', 'position_mapping', 'reverse_mapping', 
              'modified_sequence', 'symmetry', and 'applied' fields, or None if no valid mirror symmetries found
    """
    # Find mirror symmetries in the sequence
    symmetries = find_sequence_symmetries(rna_sequence, min_len)
    mirror_symmetries = symmetries['mirror_symmetries']
    
    if not mirror_symmetries:
        print(f"  No mirror symmetries found with min_len={min_len}")
        return None
    
    print(f"  Found {len(mirror_symmetries)} mirror symmetrie(s) with min_len={min_len}: {mirror_symmetries}")
    
    # Use the longest mirror symmetry for reduction
    mirror_symmetry = max(mirror_symmetries, key=lambda x: x[4])  # Sort by length (index 4)
    start, end, mirror_start, mirror_end, length = mirror_symmetry
    n = bond_matrix.shape[0]
    
    # Check if mirror symmetry meets minimum length requirement
    if length < min_len:
        return None
    
    # Define the complementary regions
    first_region = list(range(start, end + 1))
    second_region = list(range(mirror_start, mirror_end + 1))
    
    # Create position mapping
    position_mapping = {}
    reverse_mapping = {}
    new_pos = 0
    
    # Map positions before first region normally
    for i in range(start):
        position_mapping[i] = new_pos
        reverse_mapping[new_pos] = [i]
        new_pos += 1
    
    # Map first complementary region to a super-node
    for pos in first_region:
        position_mapping[pos] = new_pos
    reverse_mapping[new_pos] = first_region
    new_pos += 1
    
    # Map positions between regions normally
    for i in range(end + 1, mirror_start):
        position_mapping[i] = new_pos
        reverse_mapping[new_pos] = [i]
        new_pos += 1
    
    # Map second complementary region to a super-node
    for pos in second_region:
        position_mapping[pos] = new_pos
    reverse_mapping[new_pos] = second_region
    new_pos += 1
    
    # Map positions after second region normally
    for i in range(mirror_end + 1, n):
        position_mapping[i] = new_pos
        reverse_mapping[new_pos] = [i]
        new_pos += 1
    
    # Create reduced matrix
    reduced_size = new_pos
    reduced_matrix = np.zeros((reduced_size, reduced_size), dtype=int)
    
    # Fill reduced matrix with bond strengths
    for i_orig in range(n):
        for j_orig in range(i_orig + 1, n):
            if bond_matrix[i_orig, j_orig] > 0:
                new_i, new_j = position_mapping[i_orig], position_mapping[j_orig]
                if new_i != new_j:
                    # Check if both positions are in complementary regions
                    i_in_first_region = i_orig in first_region
                    i_in_second_region = i_orig in second_region
                    j_in_first_region = j_orig in first_region
                    j_in_second_region = j_orig in second_region
                    
                    # Enhanced bond strength for complementary regions
                    bond_strength = bond_matrix[i_orig, j_orig]
                    
                    # If both positions are in complementary regions, add strong binding
                    if ((i_in_first_region and j_in_second_region) or 
                        (i_in_second_region and j_in_first_region)):
                        bond_strength = complementarity_factor * length
                    
                    reduced_matrix[min(new_i, new_j), max(new_i, new_j)] += bond_strength
    
    # Create modified sequence
    modified_sequence = []
    i = 0
    while i < len(rna_sequence):
        if start <= i <= end:
            # First complementary region - add 'M' for mirror/complementary
            modified_sequence.append('M')
            i = end + 1
        elif mirror_start <= i <= mirror_end:
            # Second complementary region - add 'M' for mirror/complementary
            modified_sequence.append('M')
            i = mirror_end + 1
        else:
            # Outside complementary regions - add normally
            modified_sequence.append(rna_sequence[i])
            i += 1
    
    modified_sequence = ''.join(modified_sequence)
    
    return {
        'reduced_matrix': reduced_matrix,
        'position_mapping': position_mapping,
        'reverse_mapping': reverse_mapping,
        'modified_sequence': modified_sequence,
        'symmetry': mirror_symmetry,
        'applied': True
    }

def symmetry_reduction(bond_matrix, rna_sequence, min_palindrome_len=10, min_repeat_len=4, min_mirror_len=3):
    """
    This function is used to reduce the problem size by leveraging the symmetry of the sequence.
    
    Args:
        bond_matrix (:class: `numpy.ndarray`):
            Original bond matrix.
        rna_sequence (str):
            The RNA sequence string.
        min_palindrome_len (int):
            Minimum length for palindromes to be considered for reduction (default: 10).
        min_repeat_len (int):
            Minimum length for repeats to be considered for reduction (default: 4).
        min_mirror_len (int):
            Minimum length for mirror symmetries to be considered for reduction (default: 3).
            
    Returns:
        dict: Dictionary containing all symmetry reduction results.
    """
    # Apply each type of symmetry reduction
    palindrome_result = reduce_palindrome_symmetry(bond_matrix, rna_sequence, min_len=min_palindrome_len)
    repeat_result = reduce_repeat_symmetry(bond_matrix, rna_sequence, min_len=min_repeat_len)
    mirror_result = reduce_mirror_symmetry(bond_matrix, rna_sequence, min_len=min_mirror_len)
    
    # Collect results
    palindrome_results = [palindrome_result] if palindrome_result else []
    repeat_results = [repeat_result] if repeat_result else []
    mirror_results = [mirror_result] if mirror_result else []
    
    # Calculate total reductions
    total_reductions = len(palindrome_results) + len(repeat_results) + len(mirror_results)
    
    return {
        'total_reductions': total_reductions,
        'palindrome_results': palindrome_results,
        'repeat_results': repeat_results,
        'mirror_results': mirror_results
    }

def merge_idle_sequence(bond_matrix, rna_sequence, min_len=5):
    """ Merges all idle sequences of consecutive nucleotides of the same type longer than min_len to super nodes.
    
    Args:
        bond_matrix (:class: `numpy.ndarray`):
            Original bond matrix of 0's and 1's.
        rna_sequence (str):
            The RNA sequence string.
        min_len (int):
            Minimum length of idle sequence to merge.
            
    Returns:
        tuple: (reduced_matrix, position_mapping, reverse_mapping, modified_sequence)
            - reduced_matrix: New matrix with idle sequences as super-nodes
            - position_mapping: Maps original positions to new positions
            - reverse_mapping: Maps new positions back to original positions
            - modified_sequence: RNA sequence with merged nodes marked differently
    """

    n = bond_matrix.shape[0]
    
    # Find idle sequences (consecutive nucleotides of the same type)
    idle_sequences = []
    current_sequence = []
    current_type = None
    
    for i in range(n):
        current_nucleotide = rna_sequence[i]
        
        if current_nucleotide == current_type:
            current_sequence.append(i)
        else:
            # Check if previous sequence meets minimum length
            if len(current_sequence) >= min_len:
                idle_sequences.append(current_sequence)
            
            # Start new sequence
            current_sequence = [i]
            current_type = current_nucleotide
    
    # Don't forget the last sequence
    if len(current_sequence) >= min_len:
        idle_sequences.append(current_sequence)
    
    # Create position mapping
    position_mapping = {}
    reverse_mapping = {}
    new_pos = 0
    i = 0
    
    while i < n:
        # Check if current position is part of an idle sequence
        in_idle_sequence = False
        for seq in idle_sequences:
            if i in seq:
                # Map all positions in this idle sequence to the same position
                for pos in seq:
                    position_mapping[pos] = new_pos
                reverse_mapping[new_pos] = seq
                new_pos += 1
                # Skip to after the sequence
                i = max(seq) + 1
                in_idle_sequence = True
                break
        
        if not in_idle_sequence:
            # Map individual position
            position_mapping[i] = new_pos
            reverse_mapping[new_pos] = [i]
            new_pos += 1
            i += 1
    
    # Create reduced matrix
    reduced_size = new_pos
    reduced_matrix = np.zeros((reduced_size, reduced_size), dtype=int)
    
    # Fill the reduced matrix, summing bond strengths for merged nodes
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangular only
            if bond_matrix[i, j] > 0:
                new_i = position_mapping[i]
                new_j = position_mapping[j]
                if new_i != new_j:  # Don't create self-loops
                    # Add bond strength (will sum if multiple bonds map to same position)
                    reduced_matrix[min(new_i, new_j), max(new_i, new_j)] += bond_matrix[i, j]
    
    # Create modified sequence where merged nodes occupy one character
    modified_sequence = []
    i = 0
    while i < n:
        # Check if current position is part of an idle sequence
        in_idle_sequence = False
        for seq in idle_sequences:
            if i in seq:
                # Add a single character for the entire idle sequence
                modified_sequence.append('X')
                # Skip to after the sequence
                i = max(seq) + 1
                in_idle_sequence = True
                break
        
        if not in_idle_sequence:
            # Add individual nucleotide
            modified_sequence.append(rna_sequence[i])
            i += 1
    
    modified_sequence = ''.join(modified_sequence)
    
    return reduced_matrix, position_mapping, reverse_mapping, modified_sequence

def merge_maximal(bond_matrix, stem_dict, target_stem=None, rna_sequence=None):
    """ Merges a single maximal stem to a super node, effectively reducing the problem size.
    
    Args:
        bond_matrix (:class: `numpy.ndarray`):
            Original bond matrix of 0's and 1's.
        stem_dict (dict):
            Dictionary with maximal stems as keys.
        target_stem (tuple, optional):
            Specific maximal stem to collapse. If None, uses the first maximal stem.
            
    Returns:
        tuple: (reduced_matrix, position_mapping, reverse_mapping, modified_sequence)
            - reduced_matrix: New matrix with the target stem as a super-node
            - position_mapping: Maps original positions to new positions
            - reverse_mapping: Maps new positions back to original positions
            - modified_sequence: RNA sequence with merged stem marked differently
    """
    
    n = bond_matrix.shape[0]
    maximal_stems = list(stem_dict.keys())

    if len(maximal_stems) == 0:
        return bond_matrix.copy(), {i: i for i in range(n)}, {i: [i] for i in range(n)}
    
    # Select which stem to collapse
    if target_stem is None:
        # Choose the longest maximal stem
        target_stem = max(maximal_stems, key=lambda stem: stem[1] - stem[0] + 1)
    
    if target_stem not in maximal_stems:
        raise ValueError(f"Target stem {target_stem} is not a maximal stem")
    
    # Create position mapping: original positions -> new positions
    position_mapping = {}
    reverse_mapping = {}
    new_pos = 0
    
    # Get positions for each side of the target stem
    first_side_positions = set()
    second_side_positions = set()
    
    for i in range(target_stem[0], target_stem[1] + 1):  # First side of stem
        first_side_positions.add(i)
    for i in range(target_stem[2], target_stem[3] + 1):  # Second side of stem
        second_side_positions.add(i)
    
    # Create position mapping maintaining relative positions
    new_pos = 0
    i = 0
    
    while i < n:
        if i in first_side_positions:
            # Map all positions in first side to the same position
            for pos in first_side_positions:
                position_mapping[pos] = new_pos
            reverse_mapping[new_pos] = list(first_side_positions)
            new_pos += 1
            # Skip to after the first side
            i = max(first_side_positions) + 1
        elif i in second_side_positions:
            # Map all positions in second side to the same position
            for pos in second_side_positions:
                position_mapping[pos] = new_pos
            reverse_mapping[new_pos] = list(second_side_positions)
            new_pos += 1
            # Skip to after the second side
            i = max(second_side_positions) + 1
        else:
            # Map individual position
            position_mapping[i] = new_pos
            reverse_mapping[new_pos] = [i]
            new_pos += 1
            i += 1
    
    # Create reduced matrix
    reduced_size = new_pos
    reduced_matrix = np.zeros((reduced_size, reduced_size), dtype=int)
    
    # Fill the reduced matrix based on position mapping
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangular only
            if bond_matrix[i, j] > 0:
                new_i = position_mapping[i]
                new_j = position_mapping[j]
                if new_i != new_j:  # Don't create self-loops
                    reduced_matrix[min(new_i, new_j), max(new_i, new_j)] += bond_matrix[i, j]

    # Create modified sequence where merged stem positions are compressed
    if rna_sequence is None:
        # If no sequence provided, return None for modified_sequence
        modified_sequence = None
    else:
        # Create compressed sequence where merged stem positions become single nodes
        modified_sequence = []
        i = 0
        
        while i < len(rna_sequence):
            # Check if current position is part of the target stem
            if (target_stem[0] <= i <= target_stem[1]) or (target_stem[2] <= i <= target_stem[3]):
                # Add a single 'X' for the entire stem
                modified_sequence.append('x')
                # Skip to after the stem
                if target_stem[0] <= i <= target_stem[1]:
                    i = target_stem[1] + 1  # Skip first side
                else:
                    i = target_stem[3] + 1  # Skip second side
            else:
                # Add individual nucleotide
                modified_sequence.append(rna_sequence[i])
                i += 1
        
        modified_sequence = ''.join(modified_sequence)
    
    return reduced_matrix, position_mapping, reverse_mapping, modified_sequence

def text_to_matrix(file_name, min_loop):
    """ Reads properly formatted RNA text file and returns a matrix of possible hydrogen bonding pairs.

    Args:
        file_name (str):
            Path to text file.
        min_loop (int):
            Minimum number of nucleotides separating two sides of a stem.

    Returns:
        :class: `numpy.ndarray`:
            Numpy matrix of 0's and 1's, where 1 represents a possible bonding pair.
    """

    # Requires text file of RNA data written in same format as examples.
    with open(file_name) as f:
        rna = "".join(("".join(line.split()[1:]) for line in f.readlines())).lower()

    # Create a dictionary of all indices where each nucleotide occurs.
    index_dict = defaultdict(list)

    # Create a dictionary giving list of indices for each nucleotide.
    for i, nucleotide in enumerate(rna):
        index_dict[nucleotide].append(i)

    # List of possible hydrogen bonds for stems.
    # Recall that 't' is sometimes used as a stand-in for 'u'.
    hydrogen_bonds = [('a', 't'), ('a', 'u'), ('c', 'g'), ('g', 't'), ('g', 'u')]

    # Create a upper triangular matrix indicating bonding pairs.
    # All bonds have strength 1 for simplicity.
    bond_matrix = np.zeros((len(rna), len(rna)), dtype=int)
    for pair in hydrogen_bonds:
        for bond in product(index_dict[pair[0]], index_dict[pair[1]]):
            if abs(bond[0] - bond[1]) > min_loop:
                bond_matrix[min(bond), max(bond)] = 1

    return bond_matrix

def make_stem_dict(bond_matrix, min_stem, min_loop):
    """ Takes a matrix of potential hydrogen binding pairs and returns a dictionary of possible stems.

    The stem dictionary records the maximal stems (under inclusion) as keys,
    where each key maps to a list of the associated stems weakly contained within the maximal stem.
    Recording stems in this manner allows for faster computations.

    Args:
        bond_matrix (:class: `numpy.ndarray`):
            Numpy matrix of 0's and 1's, where 1 represents a possible bonding pair.
        min_stem (int):
            Minimum number of nucleotides in each side of a stem.
        min_loop (int):
            Minimum number of nucleotides separating two sides of a stem.

    Returns:
        dict: Dictionary of all possible stems with maximal stems as keys.
    """

    stem_dict = {}
    n = bond_matrix.shape[0]
    # Create a copy to avoid modifying the original matrix
    working_matrix = bond_matrix.copy()

    # Iterate through matrix looking for possible stems.
    for i in range(n + 1 - (2 * min_stem + min_loop)):
        for j in range(i + 2 * min_stem + min_loop - 1, n):
            if working_matrix[i, j] > 0:
                k = working_matrix[i, j]  # Start with the bond strength at current position
                # Check down and left for length of stem.
                # Note that working_matrix is strictly upper triangular, so loop will terminate.
                offset = 1
                while i + offset < n and j - offset >= 0 and working_matrix[i + offset, j - offset] > 0:
                    k += working_matrix[i + offset, j - offset]  # Add bond strength to sum
                    working_matrix[i + offset, j - offset] = 0
                    offset += 1

                if k >= min_stem:
                    # A 4-tuple is used to represent the stem.
                    stem_dict[(i, i + offset - 1, j - offset + 1, j)] = []

    # Iterate through all sub-stems weakly contained in a maximal stem under inclusion.
    for stem in stem_dict.keys():
        stem_dict[stem].extend([(stem[0] + i, stem[0] + k, stem[3] - k, stem[3] - i)
                                for i in range(stem[1] - stem[0] - min_stem + 2)
                                for k in range(i + min_stem - 1, stem[1] - stem[0] + 1)])

    return stem_dict

def apply_reduction_method(bond_matrix, stem_dict, rna_sequence, method='idle_sequence', **kwargs):
    """ Applies the specified reduction method to the RNA folding problem.
    
    Args:
        bond_matrix (:class: `numpy.ndarray`):
            Original bond matrix.
        stem_dict (dict):
            Dictionary with maximal stems as keys.
        rna_sequence (str):
            The RNA sequence string.
        method (str):
            Reduction method to use. Options:
            - 'idle_sequence': Merge consecutive same-type nucleotides
            - 'maximal_stem': Merge the longest maximal stem
        **kwargs:
            Additional arguments for specific methods:
            - For 'idle_sequence': min_len (default=10)
            - For 'maximal_stem': target_stem (default=None)
            
    Returns:
        tuple: (reduced_matrix, position_mapping, reverse_mapping, modified_sequence)
            - reduced_matrix: New matrix after reduction
            - position_mapping: Maps original positions to new positions
            - reverse_mapping: Maps new positions back to original positions
            - modified_sequence: RNA sequence with merged nodes marked
    """
    
    if method == 'idle_sequence':
        print("Testing idle_sequence reduction:", kwargs.get('min_len', 5))
        min_len = kwargs.get('min_len', 5)
        return merge_idle_sequence(bond_matrix, rna_sequence, min_len)
    
    elif method == 'maximal_stem':
        print("Testing maximal_stem reduction:", kwargs.get('target_stem', None))
        target_stem = kwargs.get('target_stem', None)
        return merge_maximal(bond_matrix, stem_dict, target_stem, rna_sequence)
    
    else:
        raise ValueError(f"Unknown reduction method: {method}. Available methods: 'idle_sequence', 'maximal_stem'")

def solve_reduced_problem(bond_matrix, stem_dict, rna_sequence, reduction_method='maximal_stem', 
                         min_stem=3, min_loop=2, c=0.3, verbose=True, **kwargs):
    """ Complete pipeline: reduce problem, solve with D-Wave, convert back, and plot results.
    
    Args:
        bond_matrix (:class: `numpy.ndarray`):
            Original bond matrix.
        stem_dict (dict):
            Dictionary with maximal stems as keys.
        rna_sequence (str):
            The RNA sequence string.
        reduction_method (str):
            Reduction method to use ('maximal_stem' or 'idle_sequence').
        min_stem (int):
            Minimum stem length for stem detection.
        min_loop (int):
            Minimum loop size for stem detection.
        c (float):
            Pseudoknot penalty coefficient.
        verbose (bool):
            Whether to print detailed information.
        **kwargs:
            Additional arguments for specific reduction methods.
            
    Returns:
        tuple: (original_solution_stems, reduced_solution_stems, modified_sequence)
            - original_solution_stems: Solution in original coordinates
            - reduced_solution_stems: Solution in reduced coordinates
            - modified_sequence: Modified sequence with merged nodes
    """
    
    if verbose:
        print(f"\n=== Solving with {reduction_method} reduction ===")
    
    # Step 1: Apply reduction method
    reduced_matrix, pos_mapping, reverse_mapping, modified_sequence = apply_reduction_method(
        bond_matrix, stem_dict, rna_sequence, method=reduction_method, **kwargs
    )
    
    if verbose:
        print(f"Original matrix size: {bond_matrix.shape[0]}")
        print(f"Reduced matrix size: {reduced_matrix.shape[0]}")
        print(f"Reduction: {bond_matrix.shape[0]} -> {reduced_matrix.shape[0]} nodes")
        print(f"Modified sequence: {modified_sequence}")
    
    # Step 2: Create stem dictionary for reduced problem
    reduced_stem_dict = make_stem_dict(reduced_matrix, min_stem, min_loop)
    
    if not reduced_stem_dict:
        print("No stems found in reduced problem!")
        return [], [], modified_sequence
    
    if verbose:
        print(f"Found {len(reduced_stem_dict)} maximal stems in reduced problem")
    
    # Step 3: Build CQM for reduced problem
    if verbose:
        print("Building CQM for reduced problem...")
    
    cqm = build_cqm(reduced_stem_dict, min_stem, c)
    
    # Step 4: Solve with D-Wave
    if verbose:
        print("Connecting to D-Wave Solver...")
    
    try:
        from dwave.system import LeapHybridCQMSampler
        sampler = LeapHybridCQMSampler()
        
        if verbose:
            print("Finding solution...")
        
        sample_set = sampler.sample_cqm(cqm)
        sample_set.resolve()
        
        if verbose:
            print("Processing solution...")
        
        # Step 5: Process solution
        reduced_solution_stems = process_cqm_solution(sample_set, verbose)
        
        if not reduced_solution_stems:
            print("No solution found!")
            return [], [], modified_sequence
        
        # Step 6: Convert back to original coordinates
        original_solution_stems = convert_reduced_solution_to_original(reverse_mapping, reduced_solution_stems)
        
        if verbose:
            print(f"Reduced solution stems: {reduced_solution_stems}")
            print(f"Original solution stems: {original_solution_stems}")
        
        # Step 7: Create plots
        if verbose:
            print("Creating plots...")
        
        # Plot original solution
        make_plot(rna_sequence, original_solution_stems, f'{reduction_method}_original_solution', seed=50)
        
        # Plot reduced solution
        make_plot(modified_sequence, reduced_solution_stems, f'{reduction_method}_reduced_solution', seed=60)
        
        if verbose:
            print(f"Plots saved as {reduction_method}_original_solution.png and {reduction_method}_reduced_solution.png")
        
        return original_solution_stems, reduced_solution_stems, modified_sequence
        
    except ImportError:
        print("D-Wave Ocean SDK not available. Using dummy solution for demonstration.")
        # Create a dummy solution for demonstration
        dummy_stems = list(reduced_stem_dict.keys())[:2]  # Take first 2 stems as dummy solution
        original_solution_stems = convert_reduced_solution_to_original(reverse_mapping, dummy_stems)
        
        if verbose:
            print(f"Dummy reduced solution stems: {dummy_stems}")
            print(f"Dummy original solution stems: {original_solution_stems}")
        
        # Create plots
        make_plot(rna_sequence, original_solution_stems, f'{reduction_method}_original_solution', seed=50)
        make_plot(modified_sequence, dummy_stems, f'{reduction_method}_reduced_solution', seed=60)
        
        return original_solution_stems, dummy_stems, modified_sequence

def create_modified_sequence_file(modified_sequence, base_filename='modified_sequence'):
    """ Creates a temporary file with the modified sequence for plotting purposes.
    
    Args:
        modified_sequence (str):
            The modified RNA sequence string.
        base_filename (str):
            Base name for the temporary file.
            
    Returns:
        str: Path to the created temporary file.
    """
    import tempfile
    import os
    
    # Create temporary file with modified sequence
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, prefix=base_filename + '_') as temp_file:
        # Write the modified sequence in the same format as the original file
        temp_file.write(f"0 {modified_sequence}\n")
        temp_file_path = temp_file.name
    
    return temp_file_path

def convert_reduced_solution_to_original(reverse_mapping, reduced_solution_stems):
    """ Converts solution stems from reduced matrix back to original matrix using reverse mapping.
    
    Args:
        reverse_mapping (dict):
            Maps new positions back to original positions.
        reduced_solution_stems (list):
            List of stems in solution from reduced matrix, encoded as 4-tuples.
            
    Returns:
        list: List of stems in original matrix coordinates.
    """
    original_solution_stems = []
    
    for stem in reduced_solution_stems:
        # Get the mapped position groups for each side of the stem
        first_side_group = reverse_mapping[stem[0]]
        first_side_end_group = reverse_mapping[stem[1]]
        second_side_group = reverse_mapping[stem[2]]
        second_side_end_group = reverse_mapping[stem[3]]
        
        # Calculate the maximal valid length based on position ranges
        # Get the maximum possible ranges for each side
        max_first_range = reverse_mapping[stem[1]][-1] - reverse_mapping[stem[0]][0] + 1
        max_second_range = reverse_mapping[stem[3]][-1] - reverse_mapping[stem[2]][0] + 1
        
        # The stem length is limited by the smaller of the two ranges
        max_stem_length = min(max_first_range, max_second_range)
        
        # Create the stem using the maximal valid length
        first_side_start = reverse_mapping[stem[0]][0]
        first_side_end = first_side_start + max_stem_length - 1
        second_side_start = reverse_mapping[stem[2]][0]
        second_side_end = second_side_start + max_stem_length - 1
        
        # Create the stem in original coordinates
        original_stem = (first_side_start, first_side_end, second_side_start, second_side_end)
        original_solution_stems.append(original_stem)
    
    return original_solution_stems

def check_overlap(stem1, stem2):
    """ Checks if 2 stems use any of the same nucleotides.

    Args:
        stem1 (tuple):
            4-tuple containing stem information.
        stem2 (tuple):
            4-tuple containing stem information.

    Returns:
         bool: Boolean indicating if the two stems overlap.
    """

    # Check for string dummy variable used when implementing a discrete variable.
    if type(stem1) == str or type(stem2) == str:
        return False

    # Check if any endpoints of stem2 overlap with stem1.
    for val in stem2:
        if stem1[0] <= val <= stem1[1] or stem1[2] <= val <= stem1[3]:
            return True
    # Check if endpoints of stem1 overlap with stem2.
    # Do not need to check all stem1 endpoints.
    for val in stem1[1:3]:
        if stem2[0] <= val <= stem2[1] or stem2[2] <= val <= stem2[3]:
            return True

    return False

def pseudoknot_terms(stem_dict, min_stem=3, c=0.3):
    """ Creates a dictionary with all possible pseudoknots as keys and appropriate penalties as values.

    The penalty is the parameter c times the product of the lengths of the two stems in the knot.

    Args:
        stem_dict (dict):
            Dictionary with maximal stems as keys and list of weakly contained sub-stems as values.
        min_stem (int):
            Smallest number of consecutive bonds to be considered a stem.
        c (float):
            Parameter factor of the penalty on pseudoknots.

    Returns:
         dict: Dictionary with all possible pseudoknots as keys and appropriate penalty as as value pair.
    """

    pseudos = {}
    # Look within all pairs of maximal stems for possible pseudoknots.
    for stem1, stem2 in product(stem_dict.keys(), stem_dict.keys()):
        # Using product instead of combinations allows for short asymmetric checks.
        if stem1[0] + 2 * min_stem < stem2[1] and stem1[2] + 2 * min_stem < stem2[3]:
            pseudos.update({(substem1, substem2): c * (1 + substem1[1] - substem1[0]) * (1 + substem2[1] - substem2[0])
                            for substem1, substem2
                            in product(stem_dict[stem1], stem_dict[stem2])
                            if substem1[1] < substem2[0] and substem2[1] < substem1[2] and substem1[3] < substem2[2]})
    return pseudos

def make_plot(rna_sequence, stems, fig_name='RNA_plot', seed=10, bond_matrix=None, node_size_scale=1.0, edge_width_scale=1.0, dpi=300, figsize=(12, 8)):
    """ Produces graph plot and saves as .png file.

    Args:
        rna_sequence (str):
            RNA sequence string.
        stems (list):
            List of stems in solution, encoded as 4-tuples.
        fig_name (str):
            Name of file created to save figure. ".png" is added automatically
        seed (int):
            Random seed for reproducible layouts
        bond_matrix (numpy.ndarray, optional):
            Bond matrix to determine edge strengths for visualization
        node_size_scale (float):
            Scale factor for node sizes (default: 1.0)
        edge_width_scale (float):
            Scale factor for edge widths (default: 1.0)
        dpi (int):
            Resolution in dots per inch (default: 300)
        figsize (tuple):
            Figure size in inches (width, height) (default: (12, 8))
    """

    # Create plots directory if it doesn't exist
    import os
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created plots directory: {plots_dir}")

    # Clear the current figure to prevent overlapping plots
    plt.clf()
    plt.close('all')

    # Create figure with specified size and DPI
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Use the provided RNA sequence directly
    rna = rna_sequence.lower()
    


    # Create graph with edges from RNA sequence and stems. Nodes are temporarily labeled by integers.
    G = nx.Graph()
    rna_edges = [(i, i + 1) for i in range(len(rna) - 1)]
    
    # Filter stem edges to only include valid nodes within sequence bounds
    valid_stem_edges = []
    for stem in stems:
        start1, end1, start2, end2 = stem
        for i in range(stem[1] - stem[0] + 1):
            node1 = stem[0] + i
            node2 = stem[3] - i
            # Only add edge if both nodes are within sequence bounds
            if 0 <= node1 < len(rna) and 0 <= node2 < len(rna):
                valid_stem_edges.append((node1, node2))
    
    G.add_edges_from(rna_edges + valid_stem_edges)
    
    # Use the filtered stem edges for the rest of the function
    stem_edges = valid_stem_edges

    # Assign each nucleotide to a color and size.
    color_map = []
    node_sizes = {}
    base_size = 200 * node_size_scale
    merged_size = 400 * node_size_scale
    
    for i, node in enumerate(rna):
        if node == 'u':  # Merged node
            color_map.append('tab:blue')
            node_sizes[i] = base_size
        elif node == 'g':
            color_map.append('tab:red')
            node_sizes[i] = base_size
        elif node == 'c':
            color_map.append('tab:green')
            node_sizes[i] = base_size
        elif node == 'a':
            color_map.append('y')
            node_sizes[i] = base_size
        else:
            color_map.append('purple')  # Distinct color for merged nodes
            node_sizes[i] = merged_size      # Larger size for merged nodes
            

    # Get the actual nodes in the graph and create color_map and node_sizes lists
    graph_nodes = list(G.nodes())
    color_map_list = [color_map[i] if i < len(color_map) else 'tab:blue' for i in graph_nodes]
    node_sizes_list = [node_sizes.get(i, base_size) for i in graph_nodes]

    options = {"edgecolors": "tab:gray", "alpha": 0.8}
    
    # Create a weighted graph where sequence edges have higher weight than stem edges
    weighted_G = G.copy()
    
    # Set weights for different edge types
    for edge in weighted_G.edges():
        if edge in rna_edges:
            weighted_G[edge[0]][edge[1]]['weight'] = 3.0  
        elif edge in stem_edges:
            weighted_G[edge[0]][edge[1]]['weight'] = 0.5
        else:
            weighted_G[edge[0]][edge[1]]['weight'] = 1.0
    
    # Uncomment the line below to see the difference without weights:
    # weighted_G = G.copy()  # This would make all edges equal weight
    
    # Use spring layout with edge weights
    pos = nx.spring_layout(weighted_G, 
                          iterations=5000,  # Number of iterations
                          k=5.0,           # Optimal distance between nodes
                          seed=seed,       # For reproducible results
                          scale=2.0,       # Scale factor for positions
                          weight='weight') # Use edge weights
    
    # pos = nx.spring_layout(G, iterations=5000, seed=seed)
    
    nx.draw_networkx_nodes(G, pos, node_color=color_map_list, node_size=node_sizes_list, **options)

    # Create labels with better handling of special characters
    labels = {}
    for i in range(len(rna)):
        char = rna[i]
        if char in ['a', 'g', 'c', 'u']:
            labels[i] = char.upper()
        elif char in ['p', 'q', 'r', 'm']:
            labels[i] = char.upper()
        elif char == 'x':
            labels[i] = 'X'
        else:
            labels[i] = char.upper()  # Fallback for any other characters
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color="whitesmoke")

    # Draw RNA backbone edges
    backbone_width = 3.0 * edge_width_scale
    nx.draw_networkx_edges(G, pos, edgelist=rna_edges, width=backbone_width, alpha=0.5)
    
    # Draw stem edges with widths based on bond strength
    if stem_edges:
        if bond_matrix is not None:
            # Create edge widths based on bond strengths
            edge_widths = []
            for edge in stem_edges:
                i, j = edge
                # Get bond strength from matrix (use min/max to get correct position)
                bond_strength = bond_matrix[min(i, j), max(i, j)]
                # Scale bond strength to width (1-8 range) and apply edge width scale
                width = max(1.0, min(8.0, bond_strength * 2.0)) * edge_width_scale
                edge_widths.append(width)
            
            # Draw each stem edge with its specific width
            for i, edge in enumerate(stem_edges):
                nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                     width=edge_widths[i], 
                                     alpha=0.7, 
                                     edge_color='tab:red')
        else:
            # Fallback: draw all stem edges with uniform width
            stem_width = 4.5 * edge_width_scale
            nx.draw_networkx_edges(G, pos, edgelist=stem_edges, 
                                 width=stem_width, alpha=0.7, edge_color='tab:red')

    # Save plot in the plots directory with high resolution
    plot_path = os.path.join(plots_dir, fig_name + '.png')
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')

    print('\nPlot of solution saved as {}'.format(plot_path))

def build_cqm(stem_dict, min_stem, c):
    """ Creates a constrained quadratic model to optimize most likely stems from a dictionary of possible stems.

    Args:
        stem_dict (dict):
            Dictionary with maximal stems as keys and list of weakly contained sub-stems as values.
        min_stem (int):
            Smallest number of consecutive bonds to be considered a stem.
        c (float):
            Parameter factor of the penalty on pseudoknots.

    Returns:
         :class:`~dimod.ConstrainedQuadraticModel`: Optimization model for RNA folding.
    """

    # Create linear coefficients of -k^2, prioritizing inclusion of long stems.
    # We depart from the reference paper in this formulation.
    linear_coeffs = {stem: -1 * (stem[1] - stem[0] + 1) ** 2 for sublist in stem_dict.values() for stem in sublist}

    # Create constraints for overlapping and and sub-stem containment.
    quadratic_coeffs = pseudoknot_terms(stem_dict, min_stem=min_stem, c=c)

    bqm = dimod.BinaryQuadraticModel(linear_coeffs, quadratic_coeffs, 'BINARY')

    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(bqm)

    # Add constraint disallowing overlapping sub-stems included in same maximal stem.
    for stem, substems in stem_dict.items():
        if len(substems) > 1:
            # Add the variable for all zeros case in one-hot constraint
            zeros = 'Null:' + str(stem)
            cqm.add_variable('BINARY', zeros)
            cqm.add_discrete(substems + [zeros], stem)

    for stem1, stem2 in combinations(stem_dict.keys(), 2):
        # Check maximal stems first.
        if check_overlap(stem1, stem2):
            # If maximal stems overlap, compare list of smaller stems.
            for stem_pair in product(stem_dict[stem1], stem_dict[stem2]):
                if check_overlap(stem_pair[0], stem_pair[1]):
                    cqm.add_constraint(dimod.quicksum([dimod.Binary(stem) for stem in stem_pair]) <= 1)

    return cqm

def process_cqm_solution(sample_set, verbose=True):
    """ Processes samples from solution and prints relevant information.

    Prints information about the best feasible solution and returns a list of stems contained in solution.
    Returns solution as a list of stems rather than a binary string.

    Args:
        sample_set:
            :class:`~dimod.SampleSet`: Sample set of formed by sampling the RNA folding optimization model.
        verbose (bool):
            Boolean indicating if function should print additional information.

    Returns:
        list: List of stems included in optimal solution, encoded as 4-tuples.
    """

    # Filter for feasibility.
    feasible_samples = sample_set.filter(lambda s: s.is_feasible)
    # Check that feasible example exists.
    if not feasible_samples:
        raise Exception("All solutions infeasible. You may need to try again.")

    # Extract best feasible sample.
    solution = feasible_samples.first

    print('Best Energy:', solution.energy)

    # Extract stems with a positive indicator variable.
    bonded_stems = [stem for stem, val in solution.sample.items() if val == 1 and type(stem) == tuple]

    print('\nNumber of stems in best solution:', len(bonded_stems))
    print('Stems in best solution:', *bonded_stems)

    if verbose:
        print('\nNumber of variables (stems):', len(solution[0].keys()))

        # Find pseudoknots using product instead of combinations allows for short asymmetric checks.
        pseudoknots = [(stem1, stem2) for [stem1, stem2] in product(bonded_stems, bonded_stems)
                       if stem1[1] < stem2[0] and stem2[1] < stem1[2] and stem1[3] < stem2[2]]

        print('\nNumber of pseudoknots in best solution:', len(pseudoknots))
        if pseudoknots:
            print('Pseudoknots:', *pseudoknots)

    return bonded_stems

def print_matrix_and_stem_dict(matrix, stem_dict):
    print(' ', end=' ')
    for i in range(len(matrix)):
        print(i%10, end=' ')
    print()
    for i in range(len(matrix)):
        print(i%10, end=' ')
        for j in range(len(matrix[i])):
            if int(matrix[i][j]) == 0:
                print('□', end=' ')
            else:
                print(int(matrix[i][j]), end=' ')
        print()
    
    # Order stems by length in descending order (stem[1] - stem[0])
    sorted_stems = sorted(stem_dict.keys(), key=lambda stem: stem[1] - stem[0], reverse=True)
    for i in sorted_stems:
        print(i)

def print_reduced_matrix_info(reduced_matrix, pos_mapping, reverse_mapping, original_size, modified_sequence=None, min_stem=3, min_loop=2):
    """Helper function to print information about the reduced matrix."""
    print(f"Original matrix size: {original_size}")
    print(f"Reduced matrix size: {reduced_matrix.shape}")
    print(f"Reduction: {original_size} -> {reduced_matrix.shape[0]} nodes")
    
    if modified_sequence:
        print(f"\nModified sequence: {modified_sequence}")
        print("Note: 'P' marks first half of palindrome, 'Q' marks second half of palindrome, 'R' marks repeat regions, 'M' marks mirror/complementary regions, 'x' marks other merged positions")
    
    print("\nPosition mapping (original -> new):")
    for orig_pos, new_pos in sorted(pos_mapping.items()):
        print(f"  {orig_pos} -> {new_pos}")
    
    print("\nReverse mapping (new -> original positions):")
    for new_pos, orig_positions in sorted(reverse_mapping.items()):
        print(f"  {new_pos} -> {orig_positions}")

    print("\nReduced matrix:")
    print(' ', end=' ')
    for i in range(len(reduced_matrix)):
        print(i%10, end=' ')
    print()
    for i in range(len(reduced_matrix)):
        print(i%10, end=' ')
        for j in range(len(reduced_matrix[i])):
            if int(reduced_matrix[i][j]) == 0:
                print('□', end=' ')
            else:
                print(int(reduced_matrix[i][j]), end=' ')
        print()

    # Find and print maximal stems in the reduced matrix
    reduced_stem_dict = make_stem_dict(reduced_matrix, min_stem, min_loop)
    # Order stems by length in descending order (stem[1] - stem[0])
    sorted_reduced_stems = sorted(reduced_stem_dict.keys(), key=lambda stem: stem[1] - stem[0], reverse=True)
    print(f"\nMaximal stems in reduced matrix:")
    for stem in sorted_reduced_stems:
        print(stem)
    print(f"Number of maximal stems: {len(reduced_stem_dict)}")

def find_sequence_symmetries(rna_sequence, min_symmetry_length=3):
    """
    Identifies different types of symmetries in the RNA sequence.
    
    Args:
        rna_sequence (str): The RNA sequence string
        min_symmetry_length (int): Minimum length for a symmetry to be considered
        
    Returns:
        dict: Dictionary containing different types of symmetries found
            - 'palindromes': List of palindrome positions (start, end, length)
            - 'repeats': List of repeat positions (start1, end1, start2, end2, length)
            - 'mirror_symmetries': List of mirror symmetry positions
    """
    n = len(rna_sequence)
    symmetries = {
        'palindromes': [],
        'repeats': [],
        'mirror_symmetries': []
    }
    
    # 1. Find palindromes (sequences that read the same forward and backward)
    all_palindromes = []
    for start in range(n - min_symmetry_length + 1):
        for length in range(min_symmetry_length, n - start + 1):
            end = start + length - 1
            if end >= n:
                break
                
            # Check if this subsequence is a palindrome
            is_palindrome = True
            for i in range(length // 2):
                if rna_sequence[start + i] != rna_sequence[end - i]:
                    is_palindrome = False
                    break
            
            if is_palindrome:
                all_palindromes.append((start, end, length))
    
    # Filter to keep only maximal palindromes (non-nested)
    symmetries['palindromes'] = []
    for i, (start1, end1, length1) in enumerate(all_palindromes):
        is_maximal = True
        for j, (start2, end2, length2) in enumerate(all_palindromes):
            if i != j:  # Don't compare with self
                # Check if current palindrome is contained within another
                if start2 <= start1 and end1 <= end2 and length2 > length1:
                    is_maximal = False
                    break
        if is_maximal:
            symmetries['palindromes'].append((start1, end1, length1))
    
    # 2. Find direct repeats (same sequence appearing multiple times)
    for length in range(min_symmetry_length, n // 2 + 1):
        for start1 in range(n - 2 * length + 1):
            seq1 = rna_sequence[start1:start1 + length]
            start2 = start1 + length
            
            # Look for the same sequence later in the string
            for start2 in range(start1 + length, n - length + 1):
                seq2 = rna_sequence[start2:start2 + length]
                if seq1 == seq2:
                    symmetries['repeats'].append((start1, start1 + length - 1, 
                                                start2, start2 + length - 1, length))
    
    # 3. Find mirror symmetries (complementary sequences)
    # A-T, G-C are complementary pairs
    complement_map = {'a': 't', 't': 'a', 'g': 'c', 'c': 'g'}
    
    for start in range(n - min_symmetry_length + 1):
        for length in range(min_symmetry_length, n - start + 1):
            end = start + length - 1
            if end >= n:
                break
                
            # Look for the complementary sequence
            for mirror_start in range(start + length, n - length + 1):
                mirror_end = mirror_start + length - 1
                is_complement = True
                
                for i in range(length):
                    # Check if both positions have valid nucleotides
                    if (rna_sequence[start + i] not in complement_map or 
                        rna_sequence[mirror_start + length - 1 - i] not in complement_map):
                        is_complement = False
                        break
                    
                    # Check if they are complementary (reverse order for mirror)
                    if rna_sequence[mirror_start + length - 1 - i] != complement_map[rna_sequence[start + i]]:
                        is_complement = False
                        break
                
                if is_complement:
                    symmetries['mirror_symmetries'].append((start, end, mirror_start, mirror_end, length))
                    break
    
    return symmetries

def print_symmetry_analysis(rna_sequence, symmetries):
    """Helper function to print symmetry analysis in a readable format."""
    print(f"\nSymmetry Analysis for: {rna_sequence}")
    
    if symmetries['palindromes']:
        print("\nMaximal palindromes found:")
        for start, end, length in symmetries['palindromes']:
            palindrome = rna_sequence[start:end+1]
            print(f"  Positions {start}-{end} (length {length}): '{palindrome}'")
    else:
        print("\nNo palindromes found.")
    
    if symmetries['repeats']:
        print("\nDirect repeats found:")
        for start1, end1, start2, end2, length in symmetries['repeats']:
            repeat1 = rna_sequence[start1:end1+1]
            repeat2 = rna_sequence[start2:end2+1]
            print(f"  '{repeat1}' at positions {start1}-{end1} and '{repeat2}' at positions {start2}-{end2} (length {length})")
    else:
        print("\nNo direct repeats found.")
    
    if symmetries['mirror_symmetries']:
        print("\nMirror symmetries (complementary) found:")
        for start, end, mirror_start, mirror_end, length in symmetries['mirror_symmetries']:
            seq1 = rna_sequence[start:end+1]
            seq2 = rna_sequence[mirror_start:mirror_end+1]
            print(f"  '{seq1}' at positions {start}-{end} and '{seq2}' at positions {mirror_start}-{mirror_end} (length {length})")
    else:
        print("\nNo mirror symmetries found.")

def print_symmetry_results(symmetry_results, original_matrix_size):
    """
    Print detailed results of symmetry reduction operations.
    
    Args:
        symmetry_results (dict):
            Results from symmetry_reduction function.
        original_matrix_size (int):
            Size of the original matrix before reduction.
    """
    print(f"\n=== Symmetry Reduction Results ===")
    print(f"Total reductions applied: {symmetry_results['total_reductions']}")
    print(f"Palindrome reductions: {len(symmetry_results['palindrome_results'])}")
    print(f"Repeat reductions: {len(symmetry_results['repeat_results'])}")
    print(f"Mirror symmetry reductions: {len(symmetry_results['mirror_results'])}")
    
    # Show detailed results for each type
    if symmetry_results['palindrome_results']:
        print(f"\nPalindrome reduction results:")
        for i, result in enumerate(symmetry_results['palindrome_results']):
            print(f"  {i+1}. Symmetry: {result['symmetry']}")
            print(f"     Modified sequence: {result['modified_sequence']}")
            print(f"     Size reduction: {original_matrix_size} -> {result['reduced_matrix'].shape[0]}")
    
    if symmetry_results['repeat_results']:
        print(f"\nRepeat reduction results:")
        for i, result in enumerate(symmetry_results['repeat_results']):
            print(f"  {i+1}. Symmetry: {result['symmetry']}")
            print(f"     Modified sequence: {result['modified_sequence']}")
            print(f"     Size reduction: {original_matrix_size} -> {result['reduced_matrix'].shape[0]}")
    
    if symmetry_results['mirror_results']:
        print(f"\nMirror symmetry reduction results:")
        for i, result in enumerate(symmetry_results['mirror_results']):
            print(f"  {i+1}. Symmetry: {result['symmetry']}")
            print(f"     Modified sequence: {result['modified_sequence']}")
            print(f"     Size reduction: {original_matrix_size} -> {result['reduced_matrix'].shape[0]}")

def create_hairpin_example(reduced_sequence):
    """
    Create a nice hairpin structure example for demonstration.
    
    Args:
        reduced_sequence (str): The reduced RNA sequence.
        
    Returns:
        list: List of stems that form a nice hairpin structure.
    """
    if len(reduced_sequence) < 6:
        # For very short sequences, create a simple stem
        return [(0, 0, len(reduced_sequence)-1, len(reduced_sequence)-1)]
    
    # Look for palindrome markers P and Q
    if 'P' in reduced_sequence and 'Q' in reduced_sequence:
        p_pos = reduced_sequence.find('P')
        q_pos = reduced_sequence.find('Q')
        if p_pos != -1 and q_pos != -1 and p_pos < q_pos:
            # Create a hairpin connecting P and Q with multiple stems
            stems = []
            
            # Main hairpin stem connecting P and Q
            start1 = max(0, p_pos - 1)
            end1 = p_pos
            start2 = q_pos
            end2 = min(len(reduced_sequence) - 1, q_pos + 1)
            stems.append((start1, end1, start2, end2))
            
            # Additional stem if there's enough space
            if p_pos > 1 and q_pos < len(reduced_sequence) - 1:
                stems.append((p_pos - 2, p_pos - 1, q_pos + 1, q_pos + 2))
            

            
            return stems
    
    # Fallback: create a general hairpin structure
    mid = len(reduced_sequence) // 2
    if len(reduced_sequence) >= 8:
        # Create a hairpin with stem length 2
        return [(mid-2, mid-1, mid+1, mid+2)]
    else:
        # Create a simple stem
        return [(0, 0, len(reduced_sequence)-1, len(reduced_sequence)-1)]

def pairs_to_stems(selected_pairs):

    """Group selected base pairs (i,j) into contiguous stems.

    Args:
        selected_pairs (Iterable[Tuple[int,int]]): Set/list of chosen base pairs (i<j).

    Returns:
        list: List of stems as 4-tuples (i_start, i_end, j_start, j_end).
    """
    pair_set = set(selected_pairs)
    used = set()
    stems = []

    for (i, j) in sorted(pair_set):
        if (i, j) in used:
            continue

        # Extend outward to get outermost pair of this stack
        si, sj = i, j
        while (si - 1, sj + 1) in pair_set:
            si -= 1
            sj += 1

        # Extend inward to get innermost pair
        ei, ej = i, j
        while (ei + 1, ej - 1) in pair_set:
            ei += 1
            ej -= 1

        # Mark all pairs in this stem as used
        for t in range(ei - si + 1):
            used.add((si + t, sj - t))

        # Save as (min_i, max_i, min_j, max_j)
        stems.append((si, ei, ej, sj))

    return stems


# =============================NEW QUBO BUILDER==================================
# === Ocean CQM builder for RNA secondary structure ===========================
# Variables: one binary x_(i,j) per candidate base pair (i<j) with loop >= Lmin.
# Objective:
#   + linear: isolated-pair penalty h0, noncanonical penalty, hairpin end ΔG(L)
#   + quadratic: Turner 2004 ΔG°37 nearest-neighbor stacks (WC+GU),
#                - hairpin end cancellation with inner neighbor,
#                - AU/GU end penalties canceled by neighbor stacks
# Constraints (CQM):
#   - For each nucleotide t: sum_{pairs touching t} x_(i,j) <= 1
#   - (optional) No pseudoknots: for each crossing pair pair, x_p + x_q <= 1
#
# Notes:
# - Pair labels are tuples (i,j) and used directly as variable labels in dimod.
# - Turner stack table below is a compact subset (WC + GU) with symmetrization.
# - Hairpin penalties are provided via a callback hairpin_fn(L) you can customize.
# ============================================================================

# ---- Turner 2004 ΔG°37 NN stacks (kcal/mol): compact core set (WC + GU) ----
# Key: ((top_5to3_b1, top_5to3_b2), (bot_3to5_b1, bot_3to5_b2))

# =========================
# Helpers
# =========================

# --- Turner 2004 ΔG°37 nearest-neighbor stacks (kcal/mol): WC + GU core ---
TURNER_WC_STACKS = {
    (('A','A'), ('U','U')): -0.93,
    (('A','U'), ('U','A')): -1.10,
    (('U','A'), ('A','U')): -1.33,
    (('C','U'), ('G','A')): -2.08,
    (('C','A'), ('G','U')): -2.11,
    (('G','U'), ('C','A')): -2.24,
    (('A','C'), ('U','G')): -2.35,
    (('U','C'), ('A','G')): -2.36,
    (('G','G'), ('C','C')): -3.26,
    (('G','C'), ('C','G')): -3.42,
}
TURNER_GU_STACKS = {
    (('A','G'), ('U','U')): -0.55,
    (('A','U'), ('U','G')): -1.36,
    (('G','U'), ('U','A')): -1.41,
    (('U','G'), ('G','C')): -2.11,
    (('G','G'), ('U','C')): -1.53,
    (('G','U'), ('C','G')): -2.51,
    (('G','C'), ('U','G')): -1.27,
    (('U','G'), ('A','U')): -1.00,
    (('U','G'), ('G','U')): +0.30,
}

def _symmetrize_stacks(base_table):
    """Add reverse/flip symmetries for dinucleotide stacks."""
    def flip(d): return (d[1], d[0])
    table = dict(base_table)
    added = True
    while added:
        added = False
        for (top, bot), g in list(table.items()):
            for k in [(flip(top), bot), (top, flip(bot)), (flip(top), flip(bot)),
                      (bot, top), (flip(bot), flip(top))]:
                if k not in table:
                    table[k] = g
                    added = True
    return table

STACKS = _symmetrize_stacks({**TURNER_WC_STACKS, **TURNER_GU_STACKS})

def hairpin_T04(L: int) -> float:
    """Coarse ΔG°37(L) penalty (kcal/mol) for hairpin closure. Tweak as you wish."""
    if L < 3:
        return 1e6  # forbid impossible/very tight loops
    # short loops: stronger penalties
    table = {3: 5.2, 4: 4.9, 5: 4.5, 6: 4.2}
    if L in table:
        return table[L]
    # longer loops: gentle decay; keeps a baseline cost
    return 4.0 + 0.2 * math.log(L / 8.0)

# ---- I/O helper ----------------------------------------------------------
def load_sequence_from_sequences_txt(file_name: str) -> str:
    """Load an RNA sequence from sequences_txt/<file_name>.

    Expected format per line: ID TOKEN1 TOKEN2 ...
    All tokens after the first are concatenated across lines.

    Returns uppercase sequence consisting of A/C/G/U only.
    """
    seq_path = join(dirname(__file__), 'sequences_txt', file_name)
    pieces = []
    with open(seq_path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 2:
                pieces.append(''.join(tokens[1:]))

    seq = ''.join(pieces).upper()
    if not seq:
        raise ValueError(f"No sequence parsed from {seq_path}. Each line should be: 'ID SEQ...'")
    invalid = set(seq) - set('ACGU')
    if invalid:
        raise ValueError(f"Invalid characters in sequence: {sorted(invalid)}")
    return seq

def run_rna_folding_pipeline(seq: str, output_prefix: str = None) -> tuple:
    """Run the complete RNA folding pipeline on a sequence.
    
    Args:
        seq (str): RNA sequence string
        output_prefix (str): Prefix for output files (default: sequence name)
        
    Returns:
        tuple: (chosen_pairs, stems, cqm, meta)
    """
    cqm, meta = build_cqm_secondary_turner(
        seq,
        Lmin=3,
        forbid_pseudoknots=False,
        h0=1.0,
        hairpin_fn=hairpin_T04,
        au_end_pen=0.5,
        gu_end_pen=0.3,
        allow_noncanonical=True,
        noncanon_linear_penalty=2.0,
        pseudoknot_soft_penalty=2.0,
    )

    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm)
    sampleset.resolve()
    best = sampleset.filter(lambda s: s.is_feasible).first or sampleset.first

    chosen_pairs = [p for p, v in best.sample.items() if v == 1 and isinstance(p, tuple)]
    stems = pairs_to_stems(chosen_pairs)
    
    if output_prefix:
        make_plot(seq, stems, output_prefix)
    
    return chosen_pairs, stems, cqm, meta

def process_sequence_list(list_file: str):
    """Process multiple RNA sequences from a list file.
    
    Args:
        list_file (str): Path to file containing sequence filenames (one per line, no .txt extension)
    """
    list_path = join(dirname(__file__), list_file)
    
    with open(list_path) as f:
        filenames = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(filenames)} sequences from {list_file}")
    
    for i, filename in enumerate(filenames, 1):
        print(f"\n--- Processing {i}/{len(filenames)}: {filename} ---")
        
        try:
            # Load sequence
            seq = load_sequence_from_sequences_txt(f"{filename}.txt")
            print(f"PBD ID: {filename}")
            print(f"Sequence: {seq}")
            
            # Run pipeline
            chosen_pairs, stems, cqm, meta = run_rna_folding_pipeline(seq, f"{filename}_result")
            
            print(f"Found {len(chosen_pairs)} base pairs")
            print(f"Grouped into {len(stems)} stems")
            if stems:
                print(f"Stems: {stems}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"\nCompleted processing {len(filenames)} sequences")

# ---- Utilities ----------------------------------------------------------
def stack_dG(top1, top2, bot1, bot2):
    """Lookup Turner ΔG°37 for a 5'->3' top dinuc and 3'->5' bottom dinuc."""
    return STACKS.get(((top1, top2), (bot1, bot2)), 0.0)

def crossing(p, q):
    """Return True if pair p=(i,j) crosses q=(u,v): i<u<j<v or u<i<v<j."""
    (i, j), (u, v) = p, q
    return (i < u < j < v) or (u < i < v < j)

def enumerate_candidates(seq, Lmin, allow_noncanonical):
    """Enumerate candidate base pairs and flag noncanonicals."""
    n = len(seq)
    WC = {('A','U'), ('U','A'), ('G','C'), ('C','G')}
    wobble = {('G','U'), ('U','G')}

    def allowed(a, b):
        return (a, b) in WC or (a, b) in wobble or allow_noncanonical

    P, is_noncanon = [], {}
    for i in range(n):
        for j in range(i + 1 + Lmin, n):
            a, b = seq[i], seq[j]
            if allowed(a, b):
                P.append((i, j))
                is_noncanon[(i, j)] = ((a, b) not in WC and (a, b) not in wobble)
    return P, is_noncanon

# =========================
# CQM Builder
# =========================
def build_cqm_secondary_turner(
    seq: str,
    Lmin: int = 3,
    *,
    forbid_pseudoknots: bool = True,
    # objective components (kcal/mol; + = penalty, − = bonus)
    h0: float = 1.0,                       # isolated-pair penalty
    hairpin_fn=lambda L: 0.0,              # ΔG°37(L) at helix ends
    au_end_pen: float = 0.0,               # AU terminal end penalty (per end)
    gu_end_pen: float = 0.0,               # GU terminal end penalty (per end)
    allow_noncanonical: bool = True,
    noncanon_linear_penalty: float = 1.5,
    pseudoknot_soft_penalty: Optional[float] = None  # if allowing PKs, add to objective
):
    """
    Build a dimod.ConstrainedQuadraticModel for RNA secondary structure.

    Variables: one binary x_(i,j) per candidate pair (i<j, loop >= Lmin).
    Objective: Turner stacks (quadratic) + linear penalties/bonuses (isolated, hairpin-ends,
               AU/GU ends, noncanonical) + optional soft PK penalties.
    Hard constraints: non-overlap per nucleotide; optional no-pseudoknots.
    """
    n = len(seq)
    # 1) Candidates
    P, is_noncanon = enumerate_candidates(seq, Lmin, allow_noncanonical)
    Pset = set(P)

    # 2) Build objective as a BQM (then lift to CQM)
    lin = defaultdict(float)
    quad = defaultdict(float)

    def add_lin(p, v): lin[p] += v
    def add_quad(p, q, v):
        if v == 0.0 or p == q: return
        a, b = (p, q) if p < q else (q, p)
        quad[(a, b)] += v

    # 2a) Isolated & noncanonical penalties (linear)
    for p in P:
        add_lin(p, h0 + (noncanon_linear_penalty if is_noncanon[p] else 0.0))

    # 2b) Nearest-neighbor stacking bonuses (quadratic)
    for (i, j) in P:
        inner = (i + 1, j - 1)
        if inner in Pset:
            dG = stack_dG(seq[i], seq[i+1], seq[j], seq[j-1])  # negative = stabilizing
            add_quad((i, j), inner, dG)

    # 2c) Hairpin end penalty via cancellation with inner neighbor
    for (i, j) in P:
        L = j - i - 1
        H = hairpin_fn(L)
        if H != 0.0:
            add_lin((i, j), +H)
            inner = (i + 1, j - 1)
            if inner in Pset:
                add_quad((i, j), inner, -H)

    # 2d) AU/GU terminal-end penalties (per end), canceled by neighbor stacks
    def end_pen(a, b):
        if (a, b) in {('A','U'), ('U','A')}: return au_end_pen
        if (a, b) in {('G','U'), ('U','G')}: return gu_end_pen
        return 0.0

    for (i, j) in P:
        pen = end_pen(seq[i], seq[j])
        if pen == 0.0: continue
        # inner end
        add_lin((i, j), +pen)
        inner = (i + 1, j - 1)
        if inner in Pset:
            add_quad((i, j), inner, -pen)
        # outer end
        add_lin((i, j), +pen)
        outer = (i - 1, j + 1)
        if outer in Pset:
            add_quad((i, j), outer, -pen)

    # 2e) (Optional) soft pseudoknot penalty in the objective
    if pseudoknot_soft_penalty and pseudoknot_soft_penalty > 0:
        for a in range(len(P)):
            p = P[a]
            for b in range(a + 1, len(P)):
                q = P[b]
                if crossing(p, q):
                    add_quad(p, q, +pseudoknot_soft_penalty)

    bqm = dimod.BinaryQuadraticModel(dict(lin), dict(quad), 0.0, vartype=dimod.BINARY)

    # 3) Promote to CQM and add hard constraints
    cqm = dimod.CQM()
    cqm.set_objective(bqm)

    # 3a) Non-overlap: each nucleotide pairs at most once
    for t in range(n):
        touching = [p for p in P if p[0] == t or p[1] == t]
        if touching:
            cqm.add_constraint(dimod.quicksum(dimod.Binary(p) for p in touching) <= 1,
                               label=f"nonoverlap_t{t}")

    # 3b) (Optional) hard no-pseudoknot constraints
    if forbid_pseudoknots:
        for a in range(len(P)):
            p = P[a]
            for b in range(a + 1, len(P)):
                q = P[b]
                if crossing(p, q):
                    cqm.add_constraint(dimod.Binary(p) + dimod.Binary(q) <= 1,
                                       label=f"no_pk_{p}_{q}")

    # meta data for the model
    meta = {
        "candidates": P,
        "is_noncanonical": is_noncanon,
        "notes": {
            "objective": "Turner NN stacks + linear penalties/bonuses; negative ΔG favored",
            "constraints": f"non-overlap hard; pseudoknots {'forbidden' if forbid_pseudoknots else 'allowed/soft'}",
            "params": {"Lmin": Lmin, "h0": h0, "au_end_pen": au_end_pen,
                       "gu_end_pen": gu_end_pen, "allow_noncanonical": allow_noncanonical,
                       "noncanon_linear_penalty": noncanon_linear_penalty}
        }
    }
    return cqm, meta
# =============================NEW QUBO BUILDER END==================================


# Create command line functionality.
DEFAULT_PATH = join(dirname(__file__), 'RNA_text_files', 'NC_008516.txt')


@click.command(help='Solve an instance of the RNA folding problem using '
                    'LeapHybridCQMSampler.')
@click.option('--path', type=click.Path(), default=DEFAULT_PATH,
              help=f'Path to problem file.  Default is {DEFAULT_PATH!r}')
@click.option('--verbose/--no-verbose', default=True,
              help='Prints additional model information.')
@click.option('--min-stem', type=click.IntRange(1,), default=3,
              help='Minimum length for a stem to be considered.')
@click.option('--min-loop', type=click.IntRange(0,), default=2,
              help='Minimum number of nucleotides separating two sides of a stem.')
@click.option('-c', type=click.FloatRange(0,), default=0.3,
              help='Multiplier for the coefficient of the quadratic terms for pseudoknots.')
def main(path, verbose, min_stem, min_loop, c):
    """ Find optimal stem configuration of an RNA sequence.

    Reads file, creates constrained quadratic model, solves model, and creates a plot of the result.
    Default parameters are set by click module inputs.

    Args:
        path (str):
            Path to problem file with RNA sequence.
        verbose (bool):
            Boolean to determine amount of information printed.
        min_stem (int):
            Smallest number of consecutive bonds to be considered a stem.
        min_loop (int):
            Minimum number of nucleotides separating two sides of a stem.
        c (float):
            Multiplier for the coefficient of the quadratic terms for pseudoknots.

    Returns:
        None: None
    """
    
    # Process a single sequence
    PBD_ID = "XXXX"
    seq = load_sequence_from_sequences_txt(f"{PBD_ID}.txt")
    print(seq)
    chosen_pairs, stems, cqm, meta = run_rna_folding_pipeline(seq, f"{PBD_ID}_result")

    # Process multiple sequences from a list file
    process_sequence_list("sequence_list.txt")

    """
    #### IDLE SEQUENCE REDUCTION ####
    # Read the RNA sequence for idle sequence detection
    with open(path) as f:
        rna_sequence = "".join(("".join(line.split()[1:]) for line in f.readlines())).lower()
    
    matrix = text_to_matrix(path, min_loop)
    matrix_copy = np.copy(matrix)
    stem_dict = make_stem_dict(matrix_copy, min_stem, min_loop)

    print_matrix_and_stem_dict(matrix, stem_dict)
    """
    """
    ### Testing the reduction methods using hand made examples ###
    # Test the reduction methods using the helper function
    print("\n=== Testing reduction methods ===")
    
    reduced_matrix, pos_mapping, reverse_mapping, modified_sequence = apply_reduction_method(
        matrix, stem_dict, rna_sequence, method='idle_sequence', min_len=4
    )
    print_reduced_matrix_info(reduced_matrix, pos_mapping, reverse_mapping, matrix.shape[0], modified_sequence, min_stem, min_loop)

    # Test the conversion function with example solution stems
    print("\n=== Testing conversion function ===")

    solution_stems = [(7, 12, 75, 80), (18, 22, 62, 66), (27, 30, 56, 59), (34, 37, 46, 49)]
    print(f"Original solution stems: {solution_stems}")

    reduced_solution_stems = [(7, 11, 69, 73), (15, 17, 53, 55), (24, 27, 49, 52), (31, 33, 40, 42)]
    print(f"Reduced solution stems: {reduced_solution_stems}")
    
    # Convert back to original coordinates
    converted_solution_stems = convert_reduced_solution_to_original(reverse_mapping, reduced_solution_stems)
    print(f"Converted solution stems: {converted_solution_stems}")

    # Create plots for both original and reduced solutions
    make_plot(rna_sequence, solution_stems, '2_original_solution', seed=60, node_size_scale=0.3, edge_width_scale=0.3)
    make_plot(rna_sequence, converted_solution_stems, '2_converted_solution', seed=50, node_size_scale=0.3, edge_width_scale=0.3)
    make_plot(modified_sequence, reduced_solution_stems, '2_reduced_solution', seed=60, node_size_scale=0.3, edge_width_scale=0.3)
    """
    """
    #### MAXIMAL STEM REDUCTION ####
    # Read the RNA sequence for idle sequence detection
    with open(path) as f:
        rna_sequence = "".join(("".join(line.split()[1:]) for line in f.readlines())).lower()
    
    matrix = text_to_matrix(path, min_loop)
    matrix_copy = np.copy(matrix)
    stem_dict = make_stem_dict(matrix_copy, min_stem, min_loop)

    print_matrix_and_stem_dict(matrix, stem_dict)

    
    ### Testing the reduction methods using hand made examples ###
    # Test the reduction methods using the helper function
    print("\n=== Testing reduction methods ===")
    
    reduced_matrix, pos_mapping, reverse_mapping, modified_sequence = apply_reduction_method(
        matrix, stem_dict, rna_sequence, method='maximal_stem'
    )
    print_reduced_matrix_info(reduced_matrix, pos_mapping, reverse_mapping, matrix.shape[0], modified_sequence, min_stem, min_loop)

    # Test the conversion function with example solution stems
    print("\n=== Testing conversion function ===")

    solution_stems = [(1, 3, 13, 15), (6, 10, 20, 24)]
    print(f"Original solution stems: {solution_stems}")

    # Example solution stems from the reduced matrix
    reduced_stems = make_stem_dict(reduced_matrix, min_stem, min_loop)
    print(f"Reduced stems: {reduced_stems.keys()}")

    reduced_solution_stems = [(1, 3, 9, 11), (6, 6, 16, 16)]
    print(f"Reduced solution stems: {reduced_solution_stems}")
    
    # Convert back to original coordinates
    converted_solution_stems = convert_reduced_solution_to_original(reverse_mapping, reduced_solution_stems)
    print(f"Converted solution stems: {converted_solution_stems}")

    # Create plots for both original and reduced solutions
    make_plot(rna_sequence, solution_stems, 'original_solution', seed=50)
    make_plot(rna_sequence, converted_solution_stems, 'converted_solution', seed=50)
    make_plot(modified_sequence, reduced_solution_stems, 'reduced_solution', seed=60)
    """
    """
    #### Running the full pipeline ####
    # Test with maximal stem reduction
    print("\n--- Testing maximal stem reduction ---")
    original_stems_max, reduced_stems_max, modified_seq_max = solve_reduced_problem(
        matrix, stem_dict, rna_sequence, reduction_method='maximal_stem', 
        min_stem=min_stem, min_loop=min_loop, c=c, verbose=True
    )
    """
    


    """
    # Test with idle sequence reduction
    print("\n--- Testing idle sequence reduction ---")
    original_stems_idle, reduced_stems_idle, modified_seq_idle = solve_reduced_problem(
        matrix, stem_dict, rna_sequence, reduction_method='idle_sequence', 
        min_stem=min_stem, min_loop=min_loop, c=c, verbose=True, min_len=4
    )
    """
    
    
if __name__ == "__main__":
    main()
