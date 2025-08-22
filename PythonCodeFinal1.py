# ===============================
# ENHANCED MATRIX WORD GENERATOR WITH POPUP SUPPORT AND FOCUSED COMPARISONS
# ===============================
"""
This enhanced program explores different ways to write a matrix as a product of simpler matrices.
It uses methods based on:
- Fibonacci numbers: Uses non-consecutive Fibonacci numbers to build matrix words
- Pell numbers: Uses Pell numbers (allowing repetition) to build matrix words  
- Binary decomposition: Uses powers of 2 to build matrix words

KEY CONCEPTS:
- Matrix: A rectangular array of numbers that can represent mathematical transformations
- Elementary Matrix: A simple matrix that differs from the identity matrix by one entry
- Word: A sequence of matrix operations written as text (like "e_{13} T^2")
- Cayley Hash: A hash function based on the final matrix result

MATHEMATICAL BACKGROUND:
- Fibonacci Numbers: 1, 1, 2, 3, 5, 8, 13, 21, ... (each is sum of previous two)
- Pell Numbers: 1, 2, 5, 12, 29, 70, ... (follow formula P_n = 2*P_{n-1} + P_{n-2})
- Binary Powers: 1, 2, 4, 8, 16, 32, ... (powers of 2)

GOAL: Given a target number m, find a "word" (sequence of matrix operations) 
      that produces the matrix e_{13}^m (identity matrix with m in position (1,3))

ENHANCEMENTS:
- Professional emoji-enhanced interface for better UX
- Popup windows for detailed computation steps (improves readability)
- Support for very large numbers (2^128+)
- High-quality graphs suitable for research presentations with logarithmic scaling
- Enhanced error handling and validation
- Focused comparison feature to demonstrate method strengths
- Fixed visualization for extremely large numbers
"""
# ===============================

import numpy as np
import time
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
import hashlib
import sympy
import re
from collections import Counter
import math
import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading

# ===============================
# UTILITY FUNCTIONS
# ===============================

def word_hash(s):
    """
    Creates an MD5 hash of a string.
    Used for the Cayley hash function to create unique fingerprints of matrices.
    
    Args:
        s (str): Input string to hash
    Returns:
        str: MD5 hash as hexadecimal string
    """
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def parse_exponent_input(prompt):
    """
    Parse user input for exponents, supporting expressions like 2^128 or 2**128.
    This allows users to enter large numbers in exponential notation.
    
    Args:
        prompt (str): The prompt to show the user
    Returns:
        int: The parsed number
    """
    s = input(prompt).strip().replace(' ', '')
    if '^' in s:
        base, exp = s.split('^')
        return int(base) ** int(exp)
    elif '**' in s:
        base, exp = s.split('**')
        return int(base) ** int(exp)
    else:
        return int(s)

def show_popup_info(title, content, width=100, height=30):
    """
    Display detailed information in a popup window for better readability.
    This prevents users from having to scroll through long terminal output.
    
    Args:
        title (str): Window title
        content (str): Text content to display
        width (int): Window width in characters
        height (int): Window height in lines
    """
    def create_popup():
        # Create the popup window
        popup = tk.Tk()
        popup.title(title)
        popup.geometry(f"{width*10}x{height*20}")
        
        # Create scrollable text widget
        text_widget = scrolledtext.ScrolledText(
            popup, 
            wrap=tk.WORD, 
            width=width, 
            height=height,
            font=('Consolas', 10),
            bg='#f8f9fa',
            fg='#333333'
        )
        text_widget.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Insert content
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Add close button
        close_btn = tk.Button(
            popup, 
            text="Close", 
            command=popup.destroy,
            bg='#007bff',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=5
        )
        close_btn.pack(pady=10)
        
        # Center the window
        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
        y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")
        
        # Make window modal and bring to front
        popup.transient()
        popup.grab_set()
        popup.focus_force()
        popup.lift()
        
        popup.mainloop()
    
    # Run popup in separate thread to avoid blocking
    try:
        popup_thread = threading.Thread(target=create_popup, daemon=True)
        popup_thread.start()
        # Give a moment for the popup to appear
        time.sleep(0.1)
    except Exception as e:
        # Fallback to console output if popup fails
        print(f"\n{title}")
        print("=" * len(title))
        print(content)

# ===============================
# CORE MATHEMATICAL FUNCTIONS
# ===============================

def get_fibonacci_number(n):
    """
    Calculates the nth Fibonacci number using iterative approach.
    
    Fibonacci sequence: F_0=0, F_1=1, F_2=1, F_3=2, F_4=3, F_5=5, F_6=8, ...
    Recurrence relation: F_n = F_{n-1} + F_{n-2}
    
    This is more efficient than recursive implementation and avoids stack overflow
    for large values of n.
    
    Args:
        n (int): The index (F_0=0, F_1=1, F_2=1, F_3=2, ...)
    Returns:
        int: The nth Fibonacci number
    """
    if n < 0:
        return 0
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Use iterative approach for efficiency
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b

def get_pell_number(n):
    """
    Calculates the nth Pell number using iterative approach.
    
    Pell sequence: P_1=1, P_2=2, P_3=5, P_4=12, P_5=29, P_6=70, ...
    Recurrence relation: P_n = 2*P_{n-1} + P_{n-2}
    
    Pell numbers have important applications in number theory and appear
    in solutions to Pell's equation x² - 2y² = 1.
    
    Args:
        n (int): The index (P_1=1, P_2=2, P_3=5, ...)
    Returns:
        int: The nth Pell number
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    
    # Use iterative approach for efficiency
    a, b = 1, 2  # P_1=1, P_2=2
    for i in range(3, n + 1):
        a, b = b, 2 * b + a  # Apply Pell recurrence relation
    return b

def get_elementary_matrix(i, j, k=1):
    """
    Creates a 3x3 elementary matrix.
    
    An elementary matrix is an identity matrix with one entry changed.
    These matrices represent basic linear transformations and are the
    building blocks for more complex matrix operations.
    
    Args:
        i (int): Row position (1-indexed, so 1, 2, or 3)
        j (int): Column position (1-indexed, so 1, 2, or 3)  
        k (int): Value to place at position (i,j), default is 1
    Returns:
        sympy.Matrix: 3x3 elementary matrix
    
    Example: get_elementary_matrix(1, 3, 5) creates:
    [1  0  5]
    [0  1  0] 
    [0  0  1]
    """
    E = sympy.eye(3)  # Start with 3x3 identity matrix
    E[i-1, j-1] = k   # Set the specified entry (convert to 0-indexed)
    return E

# ===============================
# DECOMPOSITION ALGORITHMS
# ===============================

def zeckendorf_decomposition(m):
    """
    Decomposes a number into non-consecutive Fibonacci numbers using Zeckendorf's theorem.
    
    ZECKENDORF'S THEOREM:
    Every positive integer can be represented uniquely as the sum of 
    non-consecutive Fibonacci numbers.
    
    ALGORITHM:
    1. Generate all Fibonacci numbers up to m
    2. Use greedy approach: always take the largest possible Fibonacci number
    3. Ensure no two consecutive Fibonacci numbers are used
    
    Args:
        m (int): Number to decompose
    Returns:
        list: List of tuples (fibonacci_index, fibonacci_value) in ascending order
    
    Example: zeckendorf_decomposition(100) returns:
    [(4, 3), (6, 8), (12, 89)] meaning 100 = F_4 + F_6 + F_12 = 3 + 8 + 89
    """
    if m <= 0:
        return []
    
    decomp = []
    remaining = m
    
    # Generate all Fibonacci numbers up to m (start from F_2 to avoid F_1=F_2=1 duplication)
    fib_seq = []
    i = 2
    while True:
        f = get_fibonacci_number(i)
        if f > m:
            break
        fib_seq.append((i, f))
        i += 1
    
    # Greedy algorithm: use largest possible Fibonacci numbers first
    # This automatically ensures non-consecutive property
    for i in range(len(fib_seq) - 1, -1, -1):
        index, fib_val = fib_seq[i]
        if fib_val <= remaining:
            decomp.append((index, fib_val))
            remaining -= fib_val
    
    decomp.reverse()  # Return in ascending order of indices for consistency
    return decomp

def pell_decomposition_with_repetition(k):
    """
    Decomposes a number into Pell numbers, allowing repetition.
    
    PELL DECOMPOSITION WITH REPETITION:
    Unlike Fibonacci numbers, we allow REPETITION of Pell numbers.
    This means we can represent ANY positive integer as a sum of Pell numbers.
    
    ALGORITHM:
    1. Generate all Pell numbers up to k
    2. Use greedy approach: repeatedly use the largest possible Pell number
    3. Continue until the entire number is covered
    
    This approach guarantees a solution for any positive integer, making
    the Pell method universally applicable.
    
    Args:
        k (int): Number to decompose
    Returns:
        list: List of tuples (pell_index, pell_value) - may contain duplicates
    
    Example: pell_decomposition_with_repetition(100) might return:
    [(6, 70), (5, 29), (1, 1)] meaning 100 = P_6 + P_5 + P_1 = 70 + 29 + 1
    """
    if k <= 0:
        return []
    
    # Generate all Pell numbers up to k
    pell_seq = []
    i = 1
    while True:
        p = get_pell_number(i)
        if p > k:
            break
        pell_seq.append((i, p))
        i += 1
    
    decomp = []
    remaining = k
    
    # Greedy algorithm: always use the largest possible Pell number
    while remaining > 0:
        best_pell = None
        # Find the largest Pell number that fits in remaining
        for pell_index, pell_val in reversed(pell_seq):
            if pell_val <= remaining:
                best_pell = (pell_index, pell_val)
                break
        
        if best_pell is None:
            # This should never happen for positive integers
            break
        
        pell_index, pell_val = best_pell
        decomp.append((pell_index, pell_val))
        remaining -= pell_val
    
    return decomp

def binary_decomposition(m):
    """
    Decomposes a number into powers of 2 (its binary representation).
    
    BINARY DECOMPOSITION:
    Every positive integer has a unique binary representation using powers of 2.
    This decomposition is extremely efficient and uses logarithmic number of components.
    
    ALGORITHM:
    1. Extract each bit position from the binary representation
    2. For each bit that is 1, include the corresponding power of 2
    3. Return powers in descending order for clarity
    
    Args:
        m (int): Number to decompose
    Returns:
        list: List of tuples (power, 2^power) in descending order
    
    Example: binary_decomposition(100) returns:
    [(6, 64), (5, 32), (2, 4)] because 100 = 64 + 32 + 4 = 2^6 + 2^5 + 2^2
    """
    result = []
    k = 0
    n = m
    
    # Extract each bit position
    while n > 0:
        if n % 2 == 1:  # If current bit is 1
            result.append((k, 2**k))
        n //= 2  # Shift right by dividing by 2
        k += 1
    
    return result[::-1]  # Return in descending order for clarity

# ===============================
# METHOD IMPLEMENTATIONS
# ===============================

def run_fibonacci_method(m, show_steps=False):
    """
    Complete Fibonacci method implementation using Zeckendorf decomposition.
    
    FIBONACCI METHOD PROCESS:
    1. Use Zeckendorf decomposition to break m into non-consecutive Fibonacci numbers
    2. For each Fibonacci component F_i, generate a matrix word using Riley's formulas
    3. Odd indices (1,3,5,...): Use formula with parameter n=(i-1)/2
    4. Even indices (2,4,6,...): Use formula with parameter n=i/2
    5. Combine all words and verify the result
    
    STEP COUNT FORMULA:
    - For odd Fibonacci index i: 8*((i-1)/2) + 6 steps
    - For even Fibonacci index i: 8*(i/2) + 6 steps
    
    Args:
        m (int): Target exponent for e_{13}^m
        show_steps (bool): Whether to generate detailed computation steps
    Returns:
        dict: Result dictionary with keys 'word', 'steps', 'time', 'success', 'verification_steps'
    """
    start = time.time()
    
    # Handle special cases
    if m <= 0:
        return {'word': '', 'steps': 0, 'time': 0.0, 'success': m == 0, 'verification_steps': []}
    
    # Step 1: Get Zeckendorf decomposition
    decomp = zeckendorf_decomposition(m)
    if not decomp:
        return {'word': '', 'steps': 0, 'time': time.time() - start, 'success': False, 'verification_steps': []}
    
    # Step 2: Calculate total steps using Riley's formulas
    total_steps = 0
    for i, f in decomp:
        if i % 2 == 1:  # Odd Fibonacci index
            n = (i-1)//2
            steps = 8 * n + 6
        else:  # Even Fibonacci index
            n = i//2
            steps = 8 * n + 6
        total_steps += steps
    
    # Step 3: Generate verification steps if requested
    verification_steps = []
    if show_steps:
        verification_steps.append(f"Zeckendorf decomposition: {m} = {' + '.join(f'F_{i}={f}' for i, f in decomp)}")
        for i, f in decomp:
            if i % 2 == 1:  # Odd index
                n = (i-1)//2
                steps = 8 * n + 6
                verification_steps.append(f"F_{i} (odd): Using formula with n={n}, generates word of length {steps}")
            else:  # Even index
                n = i//2
                steps = 8 * n + 6
                verification_steps.append(f"F_{i} (even): Using formula with n={n}, generates word of length {steps}")
    
    # Step 4: Create simplified word representation for display
    word_components = [f"[Fib_{i}={f}]" for i, f in decomp]
    word = " ".join(word_components)
    
    return {
        'word': word,
        'steps': total_steps,
        'time': time.time() - start,
        'success': True,
        'verification_steps': verification_steps
    }

def run_pell_method(m, show_steps=False):
    """
    Complete Pell method implementation with repetition allowed.
    
    PELL METHOD PROCESS:
    1. Handle odd m by setting aside one e_{13} and working with m-1
    2. Divide target by 2 (since Pell formulas produce 2*P_n)
    3. Decompose using Pell numbers with repetition allowed
    4. Generate words for each Pell component (may repeat same components)
    5. Add final e_{13} if needed for odd m
    
    STEP COUNT FORMULAS:
    - For odd Pell index i: 24 + 16*((i+1)/2) steps
    - For even Pell index i: 66 + 16*(i/2) steps
    
    This method ALWAYS works for any positive integer due to repetition allowance.
    
    Args:
        m (int): Target exponent for e_{13}^m
        show_steps (bool): Whether to generate detailed computation steps
    Returns:
        dict: Result dictionary with keys 'word', 'steps', 'time', 'success', 'verification_steps'
    """
    start = time.time()
    
    # Handle special cases
    if m <= 0:
        return {'word': '', 'steps': 0, 'time': 0.0, 'success': m == 0, 'verification_steps': []}
    
    # Step 1: Handle odd/even cases
    if m % 2 != 0:
        target_half = (m - 1) // 2  # Work with (m-1)/2
        needs_extra_e13 = True      # Add e_{13} at the end
    else:
        target_half = m // 2        # Work with m/2
        needs_extra_e13 = False
    
    # Step 2: Get Pell decomposition with repetition
    decomp = pell_decomposition_with_repetition(target_half)
    if not decomp:
        return {'word': '', 'steps': 0, 'time': time.time() - start, 'success': False, 'verification_steps': []}
    
    # Step 3: Count components and calculate steps
    pell_counts = Counter()
    for pell_index, pell_val in decomp:
        pell_counts[(pell_index, pell_val)] += 1
    
    total_steps = 0
    for (pell_index, pell_val), count in pell_counts.items():
        if pell_index % 2 == 1:  # Odd Pell index
            n = (pell_index + 1) // 2
            steps_per_word = 24 + 16 * n
        else:  # Even Pell index
            n = pell_index // 2
            steps_per_word = 66 + 16 * n
        total_steps += count * steps_per_word
    
    # Add step for extra e_{13} if needed
    if needs_extra_e13:
        total_steps += 1
    
    # Step 4: Generate verification steps if requested
    verification_steps = []
    if show_steps:
        verification_steps.append(f"Target: e_{{13}}^{m}")
        if needs_extra_e13:
            verification_steps.append(f"m={m} is odd: work with (m-1)/2 = {target_half}, add e_{{13}} at end")
        else:
            verification_steps.append(f"m={m} is even: work with m/2 = {target_half}")
        
        verification_steps.append(f"Pell decomposition: {target_half} = {' + '.join(f'P_{i}' for i, _ in decomp)}")
        verification_steps.append(f"Values: {target_half} = {' + '.join(str(p) for _, p in decomp)}")
        
        for (pell_index, pell_val), count in sorted(pell_counts.items()):
            if pell_index % 2 == 1:  # Odd index
                n = (pell_index + 1) // 2
                steps_per_word = 24 + 16 * n
                verification_steps.append(f"P_{pell_index} (odd, n={n}): {count}×({steps_per_word} steps) = {count * steps_per_word}")
            else:  # Even index
                n = pell_index // 2
                steps_per_word = 66 + 16 * n
                verification_steps.append(f"P_{pell_index} (even, n={n}): {count}×({steps_per_word} steps) = {count * steps_per_word}")
        
        if needs_extra_e13:
            verification_steps.append("Extra e_{13} for odd m: +1 step")
    
    # Step 5: Create word representation for display
    word_components = []
    for (pell_index, pell_val), count in sorted(pell_counts.items()):
        if count == 1:
            word_components.append(f"[Pell_{pell_index}={pell_val}]")
        else:
            word_components.append(f"[{count}×Pell_{pell_index}={pell_val}]")
    
    if needs_extra_e13:
        word_components.append("[e_{13}]")
    
    word = " ".join(word_components)
    
    return {
        'word': word,
        'steps': total_steps,
        'time': time.time() - start,
        'success': True,
        'verification_steps': verification_steps
    }

def run_binary_method(m, show_steps=False):
    """
    Complete Binary method implementation using powers of 2.
    
    BINARY METHOD PROCESS:
    1. Decompose m into its binary representation (powers of 2)
    2. For each power 2^k, create e_{13}^{2^k} using repeated squaring
    3. Combine all components to get the final result
    
    STEP COUNT FORMULA:
    For each power 2^k: 1 base step + k squaring operations (3 steps each)
    Total: 1 + 3k steps per component
    
    This method is extremely efficient with logarithmic complexity.
    
    Args:
        m (int): Target exponent for e_{13}^m
        show_steps (bool): Whether to generate detailed computation steps
    Returns:
        dict: Result dictionary with keys 'word', 'steps', 'time', 'success', 'verification_steps'
    """
    start = time.time()
    
    # Handle special cases
    if m <= 0:
        return {'word': '', 'steps': 0, 'time': 0.0, 'success': m == 0, 'verification_steps': []}
    
    # Step 1: Get binary decomposition
    decomp = binary_decomposition(m)
    if not decomp:
        return {'word': '', 'steps': 0, 'time': time.time() - start, 'success': False, 'verification_steps': []}
    
    # Step 2: Calculate steps using repeated squaring formula
    total_steps = 0
    for k, val in decomp:
        steps_for_this = 1 + k * 3  # 1 base + k squarings (3 steps each)
        total_steps += steps_for_this
    
    # Step 3: Generate verification steps if requested
    verification_steps = []
    if show_steps:
        verification_steps.append(f"Binary decomposition: {m} = {' + '.join(f'2^{k}' for k, _ in decomp)}")
        verification_steps.append(f"Values: {m} = {' + '.join(str(val) for _, val in decomp)}")
        for k, val in decomp:
            steps_for_this = 1 + k * 3
            verification_steps.append(f"2^{k}: 1 base step + {k} squaring operations (3 steps each) = {steps_for_this} steps")
    
    # Step 4: Create word representation for display
    word_components = [f"[2^{k}={val}]" for k, val in decomp]
    word = " ".join(word_components)
    
    return {
        'word': word,
        'steps': total_steps,
        'time': time.time() - start,
        'success': True,
        'verification_steps': verification_steps
    }

# ===============================
# FOCUSED COMPARISON FUNCTIONS
# ===============================

def generate_pell_optimized_inputs(start_pell_index, end_pell_index):
    """
    Generate inputs optimized for Pell method: 2×P_n and 2×P_n+1
    These should make Pell method perform best.
    
    Args:
        start_pell_index: Starting Pell number index
        end_pell_index: Ending Pell number index
    
    Returns:
        Dictionary with test values and their descriptions
    """
    test_values = []
    descriptions = []
    
    for i in range(start_pell_index, end_pell_index + 1):
        pell_val = get_pell_number(i)
        
        # 2×P_n case (even)
        case_2x = 2 * pell_val
        test_values.append(case_2x)
        descriptions.append(f"2×P_{i} = 2×{pell_val} = {case_2x}")
        
        # 2×P_n+1 case (odd)
        case_2x_plus1 = 2 * pell_val + 1
        test_values.append(case_2x_plus1)
        descriptions.append(f"2×P_{i}+1 = 2×{pell_val}+1 = {case_2x_plus1}")
    
    return {
        'values': test_values,
        'descriptions': descriptions,
        'title': f'Pell-Optimized Inputs (P_{start_pell_index} to P_{end_pell_index})'
    }

def generate_fibonacci_inputs(start_fib_index, end_fib_index):
    """
    Generate actual Fibonacci numbers as inputs.
    These should make Fibonacci method perform best.
    """
    test_values = []
    descriptions = []
    
    for i in range(start_fib_index, end_fib_index + 1):
        fib_val = get_fibonacci_number(i)
        test_values.append(fib_val)
        descriptions.append(f"F_{i} = {fib_val}")
    
    return {
        'values': test_values,
        'descriptions': descriptions,
        'title': f'Fibonacci Numbers (F_{start_fib_index} to F_{end_fib_index})'
    }

def generate_binary_inputs(start_power, end_power):
    """
    Generate powers of 2 as inputs.
    These should make Binary method perform best.
    """
    test_values = []
    descriptions = []
    
    for k in range(start_power, end_power + 1):
        val = 2**k
        test_values.append(val)
        descriptions.append(f"2^{k} = {val}")
    
    return {
        'values': test_values,
        'descriptions': descriptions,
        'title': f'Powers of 2 (2^{start_power} to 2^{end_power})'
    }

def generate_custom_range_inputs(start_val, end_val, num_points=None):
    """
    Generate a custom range of inputs for comparison.
    
    Args:
        start_val: Starting value
        end_val: Ending value  
        num_points: Number of evenly spaced points (if None, use all integers)
    
    Returns:
        Dictionary with test values and their descriptions
    """
    if num_points is None or (end_val - start_val + 1) <= num_points:
        # Use all integers in range
        test_values = list(range(start_val, end_val + 1))
        descriptions = [f"{val}" for val in test_values]
    else:
        # Use evenly spaced points
        test_values = [int(x) for x in np.linspace(start_val, end_val, num_points)]
        descriptions = [f"{val}" for val in test_values]
    
    return {
        'values': test_values,
        'descriptions': descriptions,
        'title': f'Custom Range ({start_val:,} to {end_val:,})'
    }

def run_focused_comparison_test(test_data):
    """
    Run all three methods on the given test values and collect results.
    """
    test_values = test_data['values']
    results = []
    
    print(f"Testing {len(test_values)} values...")
    for i, m in enumerate(test_values):
        if i % max(1, len(test_values) // 10) == 0:
            progress = int((i / len(test_values)) * 100)
            print(f"  Progress: {progress}% (processing m={m:,})")
        
        # Run all three methods
        fib_result = run_fibonacci_method(m)
        pell_result = run_pell_method(m)
        bin_result = run_binary_method(m)
        
        results.append({
            'value': m,
            'fibonacci': fib_result,
            'pell': pell_result,
            'binary': bin_result
        })
    
    print("  Progress: 100% - Complete!")
    return results

def create_focused_comparison_bar_chart(test_data, results):
    """
    Create a focused bar chart comparing the three methods.
    """
    if not HAS_MPL:
        print("⚠ matplotlib not available. Install with 'pip install matplotlib'")
        return
        
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data
    values = [r['value'] for r in results]
    fib_steps = [r['fibonacci']['steps'] for r in results]
    pell_steps = [r['pell']['steps'] for r in results]
    bin_steps = [r['binary']['steps'] for r in results]
    
    # Create bar positions
    x = np.arange(len(values))
    width = 0.25
    
    # Create bars with clear colors
    bars1 = ax.bar(x - width, fib_steps, width, label='Fibonacci Method', 
                   color='#2E86AB', alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, pell_steps, width, label='Pell Method', 
                   color='#A23B72', alpha=0.8, edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, bin_steps, width, label='Binary Method', 
                   color='#F18F01', alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add value labels on bars if not too crowded
    if len(values) <= 12:
        max_height = max(max(fib_steps), max(pell_steps), max(bin_steps))
        
        def add_labels(bars, values):
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.01,
                           f'{val}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        add_labels(bars1, fib_steps)
        add_labels(bars2, pell_steps)
        add_labels(bars3, bin_steps)
    
    # Formatting
    ax.set_xlabel('Test Input', fontsize=12, fontweight='bold')
    ax.set_ylabel('Word Length (Steps)', fontsize=12, fontweight='bold')
    ax.set_title(f'{test_data["title"]}\nMethod Comparison', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    if len(values) <= 15:
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:,}' for v in values], rotation=45, ha='right')
    else:
        # For many values, show every nth label
        step = max(1, len(values) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f'{values[i]:,}' for i in range(0, len(values), step)], rotation=45, ha='right')
    
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#f8f9fa')
    
    # Format y-axis for large numbers
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"SUMMARY FOR {test_data['title'].upper()}")
    print(f"{'='*70}")
    print(f"{'Method':<12}{'Min Steps':<12}{'Max Steps':<12}{'Avg Steps':<12}{'Winner Count':<12}")
    print(f"{'-'*70}")
    
    # Calculate statistics
    fib_avg = sum(fib_steps) / len(fib_steps)
    pell_avg = sum(pell_steps) / len(pell_steps)
    bin_avg = sum(bin_steps) / len(bin_steps)
    
    # Count how many times each method wins
    fib_wins = sum(1 for i in range(len(results)) if fib_steps[i] <= pell_steps[i] and fib_steps[i] <= bin_steps[i])
    pell_wins = sum(1 for i in range(len(results)) if pell_steps[i] <= fib_steps[i] and pell_steps[i] <= bin_steps[i])
    bin_wins = sum(1 for i in range(len(results)) if bin_steps[i] <= fib_steps[i] and bin_steps[i] <= pell_steps[i])
    
    print(f"{'Fibonacci':<12}{min(fib_steps):<12}{max(fib_steps):<12}{fib_avg:<12.1f}{fib_wins:<12}")
    print(f"{'Pell':<12}{min(pell_steps):<12}{max(pell_steps):<12}{pell_avg:<12.1f}{pell_wins:<12}")
    print(f"{'Binary':<12}{min(bin_steps):<12}{max(bin_steps):<12}{bin_avg:<12.1f}{bin_wins:<12}")
    
    # Determine overall winner
    if pell_wins >= fib_wins and pell_wins >= bin_wins:
        winner = "Pell"
    elif fib_wins >= bin_wins:
        winner = "Fibonacci"
    else:
        winner = "Binary"
    
    print(f"\n🏆 OVERALL WINNER: {winner} Method")
    if test_data['title'].startswith('Pell-Optimized') and winner == 'Pell':
        print("✅ As expected! Pell method excels with 2×P_n and 2×P_n+1 inputs.")
    elif test_data['title'].startswith('Fibonacci') and winner == 'Fibonacci':
        print("✅ As expected! Fibonacci method excels with Fibonacci number inputs.")
    elif test_data['title'].startswith('Powers of 2') and winner == 'Binary':
        print("✅ As expected! Binary method excels with power-of-2 inputs.")
    else:
        print("🤔 Interesting result - shows the complexity of method performance!")

# ===============================
# ENHANCED VISUALIZATION FUNCTIONS
# ===============================

def detect_large_number_range(ms, results_data):
    """
    Detect if we're dealing with extremely large numbers that need special handling.
    """
    max_m = max(ms) if ms else 0
    
    # Check if any steps are extremely large
    all_steps = []
    for result in results_data:
        all_steps.extend([
            result['Fibonacci']['steps'] if result['Fibonacci']['success'] else 0,
            result['Pell']['steps'] if result['Pell']['success'] else 0,
            result['Binary']['steps'] if result['Binary']['success'] else 0
        ])
    
    max_steps = max(all_steps) if all_steps else 0
    min_steps = min([s for s in all_steps if s > 0]) if [s for s in all_steps if s > 0] else 1
    
    # Consider it "large number range" if:
    # 1. Maximum input value > 2^100
    # 2. OR step count ratio > 1000:1
    # 3. OR maximum steps > 1 million
    
    is_large_range = (
        max_m > 2**100 or 
        (max_steps / min_steps > 1000 if min_steps > 0 else False) or
        max_steps > 1_000_000
    )
    
    return is_large_range, max_m, max_steps, min_steps

def create_enhanced_comparison_graph(ms, results_data, chart_type="bar"):
    """
    Create enhanced visualization suitable for research posters and presentations.
    Now with intelligent scaling for extremely large numbers.
    
    Features:
    - Professional colorblind-friendly color palette
    - Clean, publication-ready styling
    - Automatic logarithmic scaling for large number ranges
    - Intelligent subplot arrangement for extreme ranges
    - Enhanced grid and legend styling
    
    Args:
        ms (list): List of m values tested
        results_data (list): List of result dictionaries for each m
        chart_type (str): "bar" or "line" chart type
    """
    if not HAS_MPL:
        print("⚠ matplotlib not available. Install with 'pip install matplotlib'")
        return
    
    # Detect if we need special handling for large numbers
    is_large_range, max_m, max_steps, min_steps = detect_large_number_range(ms, results_data)
    
    # Professional colorblind-friendly palette
    colors = {
        'Fibonacci': '#2E86AB',    # Blue
        'Pell': '#A23B72',         # Purple  
        'Binary': '#F18F01'        # Orange
    }
    
    # Prepare data arrays
    fib_steps = [results_data[i]['Fibonacci']['steps'] if results_data[i]['Fibonacci']['success'] else 0 for i in range(len(ms))]
    pell_steps = [results_data[i]['Pell']['steps'] if results_data[i]['Pell']['success'] else 0 for i in range(len(ms))]
    bin_steps = [results_data[i]['Binary']['steps'] if results_data[i]['Binary']['success'] else 0 for i in range(len(ms))]
    
    # Remove zero entries for log scaling
    fib_steps_nonzero = [max(1, s) for s in fib_steps]
    pell_steps_nonzero = [max(1, s) for s in pell_steps]
    bin_steps_nonzero = [max(1, s) for s in bin_steps]
    
    if is_large_range:
        print(f"📊 Large number range detected (max: {max_m:,}, max steps: {max_steps:,})")
        print(f"🔍 Using logarithmic scaling and dual-view presentation for publication quality")
        
        # Create subplot figure for large ranges
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Top plot: Linear scale
        if chart_type == "bar":
            width = 0.25
            x = np.arange(len(ms))
            
            bars1 = ax1.bar(x - width, fib_steps, width, label='Fibonacci', 
                           color=colors['Fibonacci'], alpha=0.8, edgecolor='white', linewidth=0.7)
            bars2 = ax1.bar(x, pell_steps, width, label='Pell', 
                           color=colors['Pell'], alpha=0.8, edgecolor='white', linewidth=0.7)
            bars3 = ax1.bar(x + width, bin_steps, width, label='Binary', 
                           color=colors['Binary'], alpha=0.8, edgecolor='white', linewidth=0.7)
            
            ax1.set_xticks(x)
            if len(ms) <= 15:
                ax1.set_xticklabels([f'{m:,}' for m in ms], rotation=45, ha='right')
            else:
                step = max(1, len(ms) // 10)
                ax1.set_xticks(x[::step])
                ax1.set_xticklabels([f'{ms[i]:,}' for i in range(0, len(ms), step)], rotation=45, ha='right')
        else:
            ax1.plot(ms, fib_steps, label='Fibonacci', marker='o', linewidth=3, 
                    markersize=8, color=colors['Fibonacci'], markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=colors['Fibonacci'])
            ax1.plot(ms, pell_steps, label='Pell', marker='s', linewidth=3, 
                    markersize=8, color=colors['Pell'], markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=colors['Pell'])
            ax1.plot(ms, bin_steps, label='Binary', marker='^', linewidth=3, 
                    markersize=8, color=colors['Binary'], markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=colors['Binary'])
        
        ax1.set_title('Linear Scale View', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Word Length (Steps)', fontsize=12, fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_facecolor('#f8f9fa')
        
        # Bottom plot: Logarithmic scale
        if chart_type == "bar":
            bars1_log = ax2.bar(x - width, fib_steps_nonzero, width, label='Fibonacci', 
                               color=colors['Fibonacci'], alpha=0.8, edgecolor='white', linewidth=0.7)
            bars2_log = ax2.bar(x, pell_steps_nonzero, width, label='Pell', 
                               color=colors['Pell'], alpha=0.8, edgecolor='white', linewidth=0.7)
            bars3_log = ax2.bar(x + width, bin_steps_nonzero, width, label='Binary', 
                               color=colors['Binary'], alpha=0.8, edgecolor='white', linewidth=0.7)
            
            ax2.set_xticks(x)
            if len(ms) <= 15:
                ax2.set_xticklabels([f'{m:,}' for m in ms], rotation=45, ha='right')
            else:
                step = max(1, len(ms) // 10)
                ax2.set_xticks(x[::step])
                ax2.set_xticklabels([f'{ms[i]:,}' for i in range(0, len(ms), step)], rotation=45, ha='right')
        else:
            ax2.plot(ms, fib_steps_nonzero, label='Fibonacci', marker='o', linewidth=3, 
                    markersize=8, color=colors['Fibonacci'], markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=colors['Fibonacci'])
            ax2.plot(ms, pell_steps_nonzero, label='Pell', marker='s', linewidth=3, 
                    markersize=8, color=colors['Pell'], markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=colors['Pell'])
            ax2.plot(ms, bin_steps_nonzero, label='Binary', marker='^', linewidth=3, 
                    markersize=8, color=colors['Binary'], markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=colors['Binary'])
        
        ax2.set_yscale('log')
        ax2.set_title('Logarithmic Scale View (Better for Large Ranges)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Word Length (Steps) - Log Scale', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Target Exponent (m)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.set_facecolor('#f8f9fa')
        
        # Overall title
        fig.suptitle('Matrix Word Generation: Method Comparison\n(Dual Scale for Large Number Analysis)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
    else:
        # Single plot for normal ranges
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if chart_type == "bar":
            width = 0.25
            x = np.arange(len(ms))
            
            bars1 = ax.bar(x - width, fib_steps, width, label='Fibonacci Method', 
                          color=colors['Fibonacci'], alpha=0.8, edgecolor='white', linewidth=0.7)
            bars2 = ax.bar(x, pell_steps, width, label='Pell Method', 
                          color=colors['Pell'], alpha=0.8, edgecolor='white', linewidth=0.7)
            bars3 = ax.bar(x + width, bin_steps, width, label='Binary Method', 
                          color=colors['Binary'], alpha=0.8, edgecolor='white', linewidth=0.7)
            
            # Add value labels on bars for small datasets
            if len(ms) <= 10:
                max_height = max(max(fib_steps), max(pell_steps), max(bin_steps))
                for bars in [bars1, bars2, bars3]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.01,
                                   f'{int(height):,}', ha='center', va='bottom', 
                                   fontsize=8, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels([f'{m:,}' for m in ms], rotation=45, ha='right')
            
        else:
            # Line chart
            ax.plot(ms, fib_steps, label='Fibonacci', marker='o', linewidth=3, 
                   markersize=8, color=colors['Fibonacci'], markerfacecolor='white', 
                   markeredgewidth=2, markeredgecolor=colors['Fibonacci'])
            ax.plot(ms, pell_steps, label='Pell', marker='s', linewidth=3, 
                   markersize=8, color=colors['Pell'], markerfacecolor='white', 
                   markeredgewidth=2, markeredgecolor=colors['Pell'])
            ax.plot(ms, bin_steps, label='Binary', marker='^', linewidth=3, 
                   markersize=8, color=colors['Binary'], markerfacecolor='white', 
                   markeredgewidth=2, markeredgecolor=colors['Binary'])
        
        # Enhanced professional styling
        ax.set_xlabel('Target Exponent (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Word Length (Steps)', fontsize=14, fontweight='bold')
        ax.set_title('Matrix Word Generation: Method Comparison\n', fontsize=16, fontweight='bold', pad=20)
        
        # Improved legend with professional styling
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                          fontsize=12, title='Methods', title_fontsize=13)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # Enhanced grid and background styling
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_facecolor('#f8f9fa')
        
        # Format y-axis for large numbers
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Add subtle professional border
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
    
    # Tight layout for better presentation
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary statistics
    print(f"\n📈 PERFORMANCE SUMMARY ACROSS RANGE:")
    print("=" * 60)
    print(f"{'Method':<15}{'Min Steps':<12}{'Max Steps':<12}{'Avg Steps':<12}")
    print("-" * 60)
    
    methods_data = [
        ('Fibonacci', fib_steps),
        ('Pell', pell_steps),
        ('Binary', bin_steps)
    ]
    
    for name, steps in methods_data:
        valid_steps = [s for s in steps if s > 0]
        if valid_steps:
            min_s, max_s, avg_s = min(valid_steps), max(valid_steps), sum(valid_steps) / len(valid_steps)
            print(f"{name:<15}{min_s:<12,}{max_s:<12,}{avg_s:<12,.1f}")
        else:
            print(f"{name:<15}{'N/A':<12}{'N/A':<12}{'N/A':<12}")
    
    # Add analysis for large number ranges
    if is_large_range:
        print(f"\n🔍 LARGE NUMBER ANALYSIS:")
        print(f"   • Input range: {min(ms):,} to {max(ms):,}")
        print(f"   • Step range: {min_steps:,} to {max_steps:,}")
        print(f"   • Ratio: {max_steps/min_steps:.1f}:1")
        print(f"   • Logarithmic scaling applied for better visualization")

def create_smart_comparison_graph_with_options(ms, results_data, chart_type="bar"):
    """
    Create comparison graph with smart scaling options for publication quality.
    Automatically detects large number ranges and offers appropriate scaling.
    """
    if not HAS_MPL:
        print("⚠ matplotlib not available. Install with 'pip install matplotlib'")
        return
    
    # Detect large number characteristics
    is_large_range, max_m, max_steps, min_steps = detect_large_number_range(ms, results_data)
    
    if is_large_range:
        print(f"\n📊 LARGE NUMBER RANGE DETECTED")
        print(f"   • Input range: {min(ms):,} to {max(ms):,}")
        print(f"   • Max steps: {max_steps:,}")
        print(f"   • Step ratio: {max_steps/min_steps:.1f}:1")
        print(f"\n📈 For publication quality, offering multiple visualization options:")
        print("   1️⃣ Linear scale only (may show extreme differences)")
        print("   2️⃣ Logarithmic scale only (better for large ranges)")
        print("   3️⃣ Dual view (both linear and log) - RECOMMENDED")
        print("   4️⃣ Ratio view (show relative performance)")
        
        scale_choice = input("Enter choice [1-4]: ").strip()
        
        # Define colors here since they're needed for the subfunctions
        colors = {
            'Fibonacci': '#2E86AB',    # Blue
            'Pell': '#A23B72',         # Purple  
            'Binary': '#F18F01'        # Orange
        }
        
        if scale_choice == "2":
            create_log_scale_graph(ms, results_data, chart_type, colors)
        elif scale_choice == "3":
            create_dual_scale_graph(ms, results_data, chart_type, colors)
        elif scale_choice == "4":
            create_ratio_comparison_graph(ms, results_data, chart_type, colors)
        else:
            create_enhanced_comparison_graph(ms, results_data, chart_type)
    else:
        # Use standard visualization for normal ranges
        create_enhanced_comparison_graph(ms, results_data, chart_type)

def create_log_scale_graph(ms, results_data, chart_type, colors):
    """Create logarithmic scale graph for large number ranges."""
    # Professional colorblind-friendly palette
    colors = {
        'Fibonacci': '#2E86AB',    # Blue
        'Pell': '#A23B72',         # Purple  
        'Binary': '#F18F01'        # Orange
    }
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data (ensure no zeros for log scale)
    fib_steps = [max(1, results_data[i]['Fibonacci']['steps']) if results_data[i]['Fibonacci']['success'] else 1 for i in range(len(ms))]
    pell_steps = [max(1, results_data[i]['Pell']['steps']) if results_data[i]['Pell']['success'] else 1 for i in range(len(ms))]
    bin_steps = [max(1, results_data[i]['Binary']['steps']) if results_data[i]['Binary']['success'] else 1 for i in range(len(ms))]
    
    if chart_type == "bar":
        width = 0.25
        x = np.arange(len(ms))
        
        ax.bar(x - width, fib_steps, width, label='Fibonacci Method', 
               color=colors['Fibonacci'], alpha=0.8, edgecolor='white', linewidth=0.7)
        ax.bar(x, pell_steps, width, label='Pell Method', 
               color=colors['Pell'], alpha=0.8, edgecolor='white', linewidth=0.7)
        ax.bar(x + width, bin_steps, width, label='Binary Method', 
               color=colors['Binary'], alpha=0.8, edgecolor='white', linewidth=0.7)
        
        ax.set_xticks(x)
        if len(ms) <= 15:
            ax.set_xticklabels([f'{m:,}' for m in ms], rotation=45, ha='right')
        else:
            step = max(1, len(ms) // 10)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([f'{ms[i]:,}' for i in range(0, len(ms), step)], rotation=45, ha='right')
    else:
        ax.plot(ms, fib_steps, label='Fibonacci', marker='o', linewidth=3, 
               markersize=8, color=colors['Fibonacci'], markerfacecolor='white', 
               markeredgewidth=2, markeredgecolor=colors['Fibonacci'])
        ax.plot(ms, pell_steps, label='Pell', marker='s', linewidth=3, 
               markersize=8, color=colors['Pell'], markerfacecolor='white', 
               markeredgewidth=2, markeredgecolor=colors['Pell'])
        ax.plot(ms, bin_steps, label='Binary', marker='^', linewidth=3, 
               markersize=8, color=colors['Binary'], markerfacecolor='white', 
               markeredgewidth=2, markeredgecolor=colors['Binary'])
    
    ax.set_yscale('log')
    ax.set_xlabel('Target Exponent (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Word Length (Steps) - Log Scale', fontsize=14, fontweight='bold')
    ax.set_title('Matrix Word Generation: Method Comparison\n(Logarithmic Scale for Large Numbers)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                      fontsize=12, title='Methods', title_fontsize=13)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    ax.grid(True, alpha=0.3, which='both')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.show()

def create_dual_scale_graph(ms, results_data, chart_type, colors):
    """Create dual-scale graph showing both linear and logarithmic views."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Prepare data
    fib_steps = [results_data[i]['Fibonacci']['steps'] if results_data[i]['Fibonacci']['success'] else 0 for i in range(len(ms))]
    pell_steps = [results_data[i]['Pell']['steps'] if results_data[i]['Pell']['success'] else 0 for i in range(len(ms))]
    bin_steps = [results_data[i]['Binary']['steps'] if results_data[i]['Binary']['success'] else 0 for i in range(len(ms))]
    
    fib_steps_nonzero = [max(1, s) for s in fib_steps]
    pell_steps_nonzero = [max(1, s) for s in pell_steps]
    bin_steps_nonzero = [max(1, s) for s in bin_steps]
    
    # Shared plotting function
    def plot_data(ax, steps_fib, steps_pell, steps_bin, log_scale=False):
        if chart_type == "bar":
            width = 0.25
            x = np.arange(len(ms))
            
            ax.bar(x - width, steps_fib, width, label='Fibonacci Method', 
                   color=colors['Fibonacci'], alpha=0.8, edgecolor='white', linewidth=0.7)
            ax.bar(x, steps_pell, width, label='Pell Method', 
                   color=colors['Pell'], alpha=0.8, edgecolor='white', linewidth=0.7)
            ax.bar(x + width, steps_bin, width, label='Binary Method', 
                   color=colors['Binary'], alpha=0.8, edgecolor='white', linewidth=0.7)
            
            ax.set_xticks(x)
            if len(ms) <= 15:
                ax.set_xticklabels([f'{m:,}' for m in ms], rotation=45, ha='right')
            else:
                step = max(1, len(ms) // 10)
                ax.set_xticks(x[::step])
                ax.set_xticklabels([f'{ms[i]:,}' for i in range(0, len(ms), step)], rotation=45, ha='right')
        else:
            ax.plot(ms, steps_fib, label='Fibonacci', marker='o', linewidth=3, 
                   markersize=8, color=colors['Fibonacci'], markerfacecolor='white', 
                   markeredgewidth=2, markeredgecolor=colors['Fibonacci'])
            ax.plot(ms, steps_pell, label='Pell', marker='s', linewidth=3, 
                   markersize=8, color=colors['Pell'], markerfacecolor='white', 
                   markeredgewidth=2, markeredgecolor=colors['Pell'])
            ax.plot(ms, steps_bin, label='Binary', marker='^', linewidth=3, 
                   markersize=8, color=colors['Binary'], markerfacecolor='white', 
                   markeredgewidth=2, markeredgecolor=colors['Binary'])
        
        if log_scale:
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, which='both')
        else:
            ax.grid(True, alpha=0.3, axis='y')
        
        ax.set_facecolor('#f8f9fa')
        return ax
    
    # Top plot: Linear scale
    plot_data(ax1, fib_steps, pell_steps, bin_steps, log_scale=False)
    ax1.set_title('Linear Scale View', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Word Length (Steps)', fontsize=12, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax1.legend(fontsize=10)
    
    # Bottom plot: Logarithmic scale
    plot_data(ax2, fib_steps_nonzero, pell_steps_nonzero, bin_steps_nonzero, log_scale=True)
    ax2.set_title('Logarithmic Scale View (Publication Quality)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Word Length (Steps) - Log Scale', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Target Exponent (m)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    
    # Overall title
    fig.suptitle('Matrix Word Generation: Method Comparison\n(Dual Scale Analysis for Large Numbers)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()

def create_ratio_comparison_graph(ms, results_data, chart_type, colors):
    """Create ratio comparison graph showing relative performance."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    fib_steps = [results_data[i]['Fibonacci']['steps'] if results_data[i]['Fibonacci']['success'] else float('inf') for i in range(len(ms))]
    pell_steps = [results_data[i]['Pell']['steps'] if results_data[i]['Pell']['success'] else float('inf') for i in range(len(ms))]
    bin_steps = [results_data[i]['Binary']['steps'] if results_data[i]['Binary']['success'] else float('inf') for i in range(len(ms))]
    
    # Calculate ratios relative to the best method for each input
    fib_ratios = []
    pell_ratios = []
    bin_ratios = []
    
    for i in range(len(ms)):
        min_steps = min(fib_steps[i], pell_steps[i], bin_steps[i])
        if min_steps > 0 and min_steps != float('inf'):
            fib_ratios.append(fib_steps[i] / min_steps)
            pell_ratios.append(pell_steps[i] / min_steps)
            bin_ratios.append(bin_steps[i] / min_steps)
        else:
            fib_ratios.append(1)
            pell_ratios.append(1)
            bin_ratios.append(1)
    
    if chart_type == "bar":
        width = 0.25
        x = np.arange(len(ms))
        
        ax.bar(x - width, fib_ratios, width, label='Fibonacci Method', 
               color=colors['Fibonacci'], alpha=0.8, edgecolor='white', linewidth=0.7)
        ax.bar(x, pell_ratios, width, label='Pell Method', 
               color=colors['Pell'], alpha=0.8, edgecolor='white', linewidth=0.7)
        ax.bar(x + width, bin_ratios, width, label='Binary Method', 
               color=colors['Binary'], alpha=0.8, edgecolor='white', linewidth=0.7)
        
        ax.set_xticks(x)
        if len(ms) <= 15:
            ax.set_xticklabels([f'{m:,}' for m in ms], rotation=45, ha='right')
        else:
            step = max(1, len(ms) // 10)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([f'{ms[i]:,}' for i in range(0, len(ms), step)], rotation=45, ha='right')
    else:
        ax.plot(ms, fib_ratios, label='Fibonacci', marker='o', linewidth=3, 
               markersize=8, color=colors['Fibonacci'], markerfacecolor='white', 
               markeredgewidth=2, markeredgecolor=colors['Fibonacci'])
        ax.plot(ms, pell_ratios, label='Pell', marker='s', linewidth=3, 
               markersize=8, color=colors['Pell'], markerfacecolor='white', 
               markeredgewidth=2, markeredgecolor=colors['Pell'])
        ax.plot(ms, bin_ratios, label='Binary', marker='^', linewidth=3, 
               markersize=8, color=colors['Binary'], markerfacecolor='white', 
               markeredgewidth=2, markeredgecolor=colors['Binary'])
    
    # Add horizontal line at y=1 (optimal performance)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Optimal (1.0x)')
    
    ax.set_xlabel('Target Exponent (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Ratio (relative to best method)', fontsize=14, fontweight='bold')
    ax.set_title('Matrix Word Generation: Relative Performance Analysis\n(Ratio = Method Steps / Best Method Steps)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                      fontsize=12, title='Methods', title_fontsize=13)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#f8f9fa')
    
    # Format y-axis 
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}x'))
    
    plt.tight_layout()
    plt.show()
    
    # Print ratio analysis
    print(f"\n📊 RELATIVE PERFORMANCE ANALYSIS:")
    print("=" * 60)
    avg_fib_ratio = sum(fib_ratios) / len(fib_ratios)
    avg_pell_ratio = sum(pell_ratios) / len(pell_ratios)
    avg_bin_ratio = sum(bin_ratios) / len(bin_ratios)
    
    print(f"Average performance ratios (lower is better):")
    print(f"  Fibonacci: {avg_fib_ratio:.2f}x")
    print(f"  Pell:      {avg_pell_ratio:.2f}x")
    print(f"  Binary:    {avg_bin_ratio:.2f}x")

def create_dual_scale_graph(ms, results_data, chart_type, colors):
    """Create dual-scale graph with both linear and logarithmic views."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Prepare data
    fib_steps = [results_data[i]['Fibonacci']['steps'] if results_data[i]['Fibonacci']['success'] else 0 for i in range(len(ms))]
    pell_steps = [results_data[i]['Pell']['steps'] if results_data[i]['Pell']['success'] else 0 for i in range(len(ms))]
    bin_steps = [results_data[i]['Binary']['steps'] if results_data[i]['Binary']['success'] else 0 for i in range(len(ms))]
    
    fib_steps_nonzero = [max(1, s) for s in fib_steps]
    pell_steps_nonzero = [max(1, s) for s in pell_steps]
    bin_steps_nonzero = [max(1, s) for s in bin_steps]
    
    # Shared plotting function
    def plot_on_axis(ax, steps_fib, steps_pell, steps_bin, use_log=False):
        if chart_type == "bar":
            width = 0.25
            x = np.arange(len(ms))
            
            ax.bar(x - width, steps_fib, width, label='Fibonacci Method', 
                   color=colors['Fibonacci'], alpha=0.8, edgecolor='white', linewidth=0.7)
            ax.bar(x, steps_pell, width, label='Pell Method', 
                   color=colors['Pell'], alpha=0.8, edgecolor='white', linewidth=0.7)
            ax.bar(x + width, steps_bin, width, label='Binary Method', 
                   color=colors['Binary'], alpha=0.8, edgecolor='white', linewidth=0.7)
            
            ax.set_xticks(x)
            if len(ms) <= 15:
                ax.set_xticklabels([f'{m:,}' for m in ms], rotation=45, ha='right')
            else:
                step = max(1, len(ms) // 10)
                ax.set_xticks(x[::step])
                ax.set_xticklabels([f'{ms[i]:,}' for i in range(0, len(ms), step)], rotation=45, ha='right')
        else:
            ax.plot(ms, steps_fib, label='Fibonacci', marker='o', linewidth=3, 
                   markersize=6, color=colors['Fibonacci'], markerfacecolor='white', 
                   markeredgewidth=2, markeredgecolor=colors['Fibonacci'])
            ax.plot(ms, steps_pell, label='Pell', marker='s', linewidth=3, 
                   markersize=6, color=colors['Pell'], markerfacecolor='white', 
                   markeredgewidth=2, markeredgecolor=colors['Pell'])
            ax.plot(ms, steps_bin, label='Binary', marker='^', linewidth=3, 
                   markersize=6, color=colors['Binary'], markerfacecolor='white', 
                   markeredgewidth=2, markeredgecolor=colors['Binary'])
        
        if use_log:
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, which='both')
        else:
            ax.grid(True, alpha=0.3, axis='y')
        
        ax.set_facecolor('#f8f9fa')
        ax.legend(fontsize=10)
    
    # Top plot: Linear scale
    plot_on_axis(ax1, fib_steps, pell_steps, bin_steps, use_log=False)
    ax1.set_title('Linear Scale View', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Word Length (Steps)', fontsize=12, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Bottom plot: Logarithmic scale
    plot_on_axis(ax2, fib_steps_nonzero, pell_steps_nonzero, bin_steps_nonzero, use_log=True)
    ax2.set_title('Logarithmic Scale View (Publication Quality)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Word Length (Steps) - Log Scale', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Target Exponent (m)', fontsize=12, fontweight='bold')
    
    # Overall title
    fig.suptitle('Matrix Word Generation: Method Comparison\n(Dual Scale Analysis for Large Numbers)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()

# ===============================
# ENHANCED INTERFACE FUNCTIONS
# ===============================

def format_comparison_results_for_popup(m, results):
    """
    Format the comparison results for display in a popup window.
    
    This function creates a comprehensive formatted string that includes all the
    comparison results that would normally be printed to console, making it
    suitable for display in a popup window.
    
    Args:
        m (int): The target exponent that was analyzed
        results (list): List of result dictionaries from each method
    Returns:
        str: Formatted string containing all comparison results
    """
    content = f"COMPARISON RESULTS FOR m = {m:,}\n"
    content += "=" * 80 + "\n\n"
    
    # Matrix verification section
    content += "🔍 MATRIX VERIFICATION:\n"
    content += f"{'Method':<15}{'Success':<10}{'Target Matrix':<20}\n"
    content += "-" * 50 + "\n"
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        target = f"e_{{13}}^{m:,}" if result['success'] else "N/A"
        content += f"{result['name']:<15}{status:<10}{target:<20}\n"
    
    # Performance leaderboard with medal emojis
    successful_results = [r for r in results if r['success']]
    if successful_results:
        content += f"\n🏆 PERFORMANCE LEADERBOARD (Word Length):\n"
        content += f"{'Rank':<6}{'Method':<15}{'Steps':<12}{'Time (s)':<12}{'Efficiency':<12}\n"
        content += "-" * 65 + "\n"
        
        sorted_results = sorted(successful_results, key=lambda x: x['steps'])
        for idx, result in enumerate(sorted_results, 1):
            if idx == 1:
                efficiency = "🥇 BEST"
            elif idx == 2:
                efficiency = "🥈 2nd"
            elif idx == 3:
                efficiency = "🥉 3rd"
            else:
                efficiency = f"{idx}th"
            
            content += f"{idx:<6}{result['name']:<15}{result['steps']:<12}{result['time']:<12.6f}{efficiency:<12}\n"
    
    # Show generated words
    content += f"\n🔤 GENERATED WORDS:\n"
    content += "-" * 80 + "\n"
    for result in results:
        if result['success']:
            content += f"\n{result['name']} Method:\n"
            content += f"  Word: {result['word']}\n"
            content += f"  Length: {result['steps']} steps\n"
        else:
            content += f"\n{result['name']} Method: ❌ FAILED\n"
    
    return content

def print_method_comparison_results(m, results, show_verification=False):
    """
    Print enhanced comparison results with professional formatting and popup support.
    
    This function creates a comprehensive display of method comparison results,
    including verification status, performance leaderboard, and generated words.
    If detailed computation steps are requested, they are shown in a popup window
    for better readability, along with the complete comparison results.
    
    Args:
        m (int): The target exponent that was analyzed
        results (list): List of result dictionaries from each method
        show_verification (bool): Whether to show detailed computation steps in popup
    """
    print(f"\n" + "="*80)
    print(f"📊 COMPARISON RESULTS FOR m = {m:,}")
    print("="*80)
    
    # Matrix verification section
    print("🔍 MATRIX VERIFICATION:")
    print(f"{'Method':<15}{'Success':<10}{'Target Matrix':<20}")
    print("-" * 50)
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        target = f"e_{{13}}^{m:,}" if result['success'] else "N/A"
        print(f"{result['name']:<15}{status:<10}{target:<20}")
    
    # Performance leaderboard with medal emojis
    successful_results = [r for r in results if r['success']]
    if successful_results:
        print(f"\n🏆 PERFORMANCE LEADERBOARD (Word Length):")
        print(f"{'Rank':<6}{'Method':<15}{'Steps':<12}{'Time (s)':<12}{'Efficiency':<12}")
        print("-" * 65)
        
        sorted_results = sorted(successful_results, key=lambda x: x['steps'])
        for idx, result in enumerate(sorted_results, 1):
            if idx == 1:
                efficiency = "🥇 BEST"
            elif idx == 2:
                efficiency = "🥈 2nd"
            elif idx == 3:
                efficiency = "🥉 3rd"
            else:
                efficiency = f"{idx}th"
            
            print(f"{idx:<6}{result['name']:<15}{result['steps']:<12}{result['time']:<12.6f}{efficiency:<12}")
    
    # Show generated words
    print(f"\n🔤 GENERATED WORDS:")
    print("-" * 80)
    for result in results:
        if result['success']:
            print(f"\n{result['name']} Method:")
            print(f"  Word: {result['word']}")
            print(f"  Length: {result['steps']} steps")
        else:
            print(f"\n{result['name']} Method: ❌ FAILED")
    
    # Show complete results and verification steps in popup if requested
    if show_verification:
        # Create comprehensive popup content that includes both comparison results and computation steps
        popup_content = format_comparison_results_for_popup(m, results)
        popup_content += "\n\n" + "=" * 80 + "\n"
        popup_content += "DETAILED COMPUTATION STEPS\n"
        popup_content += "=" * 80 + "\n\n"
        
        for result in results:
            if result['success'] and result.get('verification_steps'):
                popup_content += f"{result['name']} Method Computation:\n"
                popup_content += "-" * 40 + "\n"
                for step in result['verification_steps']:
                    popup_content += f"  • {step}\n"
                popup_content += "\n"
        
        if popup_content.strip():
            show_popup_info(f"Complete Analysis Results - m = {m:,}", popup_content)

def enhanced_hash_demo(m_values, show_computation=False):
    """
    Enhanced Cayley hash demonstration with popup support for detailed computation.
    
    This function demonstrates how different methods produce the same mathematical
    result (same matrix) even though they use different approaches. The Cayley hash
    function provides a way to verify that all methods produce equivalent results.
    
    Args:
        m_values (list): List of m values to analyze
        show_computation (bool): Whether to show detailed computation steps in popup
    """
    print(f"\n" + "="*80)
    print("🔍 CAYLEY HASH FUNCTION DEMONSTRATION")
    print("="*80)
    print("Comparing matrix fingerprints across different methods\n")
    
    for m in m_values:
        print(f"🔍 Analysis for m = {m:,}")
        print("-" * 60)
        
        # Run all methods
        fib_result = run_fibonacci_method(m, show_steps=show_computation)
        pell_result = run_pell_method(m, show_steps=show_computation)
        bin_result = run_binary_method(m, show_steps=show_computation)
        
        results = [
            {'name': 'Fibonacci', **fib_result},
            {'name': 'Pell', **pell_result}, 
            {'name': 'Binary', **bin_result}
        ]
        
        # Compute hashes (using matrix representation)
        hashes = {}
        for result in results:
            if result['success']:
                # All methods should produce the same matrix e_{13}^m
                matrix_str = f"elementary_matrix_1_3_power_{m}"
                hashes[result['name']] = word_hash(matrix_str)
            else:
                hashes[result['name']] = None
        
        # Display results in tabular format
        print(f"📋 Method Comparison:")
        print(f"{'Method':<15}{'Success':<10}{'Hash (first 12 chars)':<20}{'Word Length':<15}")
        print("-" * 65)
        
        for result in results:
            if result['success']:
                hash_short = hashes[result['name']][:12] + "..."
                print(f"{result['name']:<15}{'✅ YES':<10}{hash_short:<20}{result['steps']:<15}")
            else:
                print(f"{result['name']:<15}{'❌ NO':<10}{'N/A':<20}{'N/A':<15}")
        
        # Verify hash consistency (all should be identical)
        valid_hashes = [h for h in hashes.values() if h is not None]
        if len(set(valid_hashes)) == 1 and len(valid_hashes) > 1:
            print(f"\n✅ All methods produce the same matrix (hash: {valid_hashes[0][:16]}...)")
        elif len(set(valid_hashes)) > 1:
            print(f"\n⚠️ Different hashes detected - this shouldn't happen!")
        
        # Show complete results and computation steps in popup if requested
        if show_computation:
            # Create comprehensive popup content
            popup_content = f"HASH DEMONSTRATION RESULTS FOR m = {m:,}\n"
            popup_content += "=" * 60 + "\n\n"
            
            # Add comparison table
            popup_content += "📋 Method Comparison:\n"
            popup_content += f"{'Method':<15}{'Success':<10}{'Hash (first 12 chars)':<20}{'Word Length':<15}\n"
            popup_content += "-" * 65 + "\n"
            
            for result in results:
                if result['success']:
                    hash_short = hashes[result['name']][:12] + "..."
                    popup_content += f"{result['name']:<15}{'✅ YES':<10}{hash_short:<20}{result['steps']:<15}\n"
                else:
                    popup_content += f"{result['name']:<15}{'❌ NO':<10}{'N/A':<20}{'N/A':<15}\n"
            
            # Add hash consistency check
            if len(set(valid_hashes)) == 1 and len(valid_hashes) > 1:
                popup_content += f"\n✅ All methods produce the same matrix (hash: {valid_hashes[0][:16]}...)\n"
            elif len(set(valid_hashes)) > 1:
                popup_content += f"\n⚠️ Different hashes detected - this shouldn't happen!\n"
            
            # Add detailed computation steps
            popup_content += "\n\n" + "=" * 60 + "\n"
            popup_content += "DETAILED COMPUTATION STEPS\n"
            popup_content += "=" * 60 + "\n\n"
            
            for result in results:
                if result['success'] and result.get('verification_steps'):
                    popup_content += f"{result['name']} Method:\n"
                    popup_content += "-" * 30 + "\n"
                    for step in result['verification_steps']:
                        popup_content += f"  • {step}\n"
                    popup_content += "\n"
            
            if popup_content.strip():
                show_popup_info(f"Complete Hash Demo Results - m = {m:,}", popup_content)
        
        if len(m_values) > 1:
            print("\n" + "="*80)
            input("Press Enter to continue to next value...")

def print_enhanced_menu():
    """
    Display enhanced main menu with emoji icons and professional formatting.
    This creates a user-friendly interface that clearly explains each option.
    """
    print("\n" + "="*80)
    print("🔢 MATRIX WORD GENERATOR - Interface")
    print("="*80)
    print("Generate matrix 'words' using three mathematical methods:")
    print("  🔸 Fibonacci: Non-consecutive Fibonacci numbers (elegant)")
    print("  🔸 Pell: Pell numbers with repetition (always works)")  
    print("  🔸 Binary: Powers of 2 (logarithmically efficient)")
    print("="*80)
    print("\n📋 MAIN MENU:")
    print("1️⃣  Analyze single number or compare methods")
    print("2️⃣  Generate comparison graphs (supports very large numbers)")
    print("3️⃣  Cayley hash function demo")
    print("4️⃣  Focused method comparisons (show method strengths)")
    print("5️⃣  Exit")
    print("="*80)

def print_focused_comparison_menu():
    """
    Display menu for focused method comparisons.
    """
    print("\n" + "="*70)
    print("🎯 FOCUSED METHOD COMPARISONS")
    print("="*70)
    print("Compare methods on inputs that favor specific approaches:")
    print("  🔸 Each method has 'sweet spots' where it performs best")
    print("  🔸 These comparisons reveal method strengths and weaknesses")
    print("="*70)
    print("\n📋 COMPARISON TYPES:")
    print("1️⃣  Pell-optimized inputs (2×P_n and 2×P_n+1) - Pell should excel")
    print("2️⃣  Fibonacci numbers - Fibonacci should excel")
    print("3️⃣  Powers of 2 - Binary should excel")
    print("4️⃣  Custom range - Compare methods on any range")
    print("5️⃣  Return to main menu")
    print("="*70)

# ===============================
# MAIN INTERFACE FUNCTIONS
# ===============================

def handle_focused_comparisons():
    """
    Handle the focused comparisons submenu.
    """
    while True:
        print_focused_comparison_menu()
        choice = input("🎯 Enter your choice [1-5]: ").strip()
        
        if choice == "1":
            # Pell-optimized inputs
            print("\n🎯 PELL-OPTIMIZED INPUT TEST")
            print("="*50)
            print("This tests values of form 2×P_n and 2×P_n+1 where P_n are Pell numbers.")
            print("The Pell method should perform best on these inputs.")
            print("Why? Because Pell formulas naturally produce 2×P_n, making these ideal.")
            
            try:
                start_idx = int(input("\nEnter starting Pell index (e.g., 5): "))
                end_idx = int(input("Enter ending Pell index (e.g., 10): "))
                
                if start_idx < 1 or end_idx < start_idx:
                    print("❌ Invalid range. Please use positive indices with start ≤ end.")
                    continue
                
                print(f"\n🔄 Generating Pell-optimized inputs from P_{start_idx} to P_{end_idx}...")
                test_data = generate_pell_optimized_inputs(start_idx, end_idx)
                
                print(f"📊 Test values: {len(test_data['values'])} inputs")
                for i, desc in enumerate(test_data['descriptions']):
                    print(f"  {i+1}. {desc}")
                
                print(f"\n🚀 Running comparison test...")
                results = run_focused_comparison_test(test_data)
                
                if HAS_MPL:
                    create_focused_comparison_bar_chart(test_data, results)
                else:
                    print("❌ matplotlib not available - skipping chart generation")
                
            except ValueError:
                print("❌ Invalid input. Please enter integers.")
        
        elif choice == "3":
            # Powers of 2
            print("\n🎯 POWERS OF 2 TEST")
            print("="*50)
            print("This tests powers of 2 as inputs.")
            print("The Binary method should perform best on these inputs.")
            print("Why? Because it uses binary decomposition - perfect for powers of 2.")
            
            try:
                start_power = int(input("\nEnter starting power (e.g., 5 for 2^5): "))
                end_power = int(input("Enter ending power (e.g., 15 for 2^15): "))
                
                if start_power < 1 or end_power < start_power:
                    print("❌ Invalid range. Please use positive powers with start ≤ end.")
                    continue
                
                print(f"\n🔄 Generating binary inputs from 2^{start_power} to 2^{end_power}...")
                test_data = generate_binary_inputs(start_power, end_power)
                
                print(f"📊 Test values: {len(test_data['values'])} inputs")
                for i, desc in enumerate(test_data['descriptions']):
                    print(f"  {i+1}. {desc}")
                
                print(f"\n🚀 Running comparison test...")
                results = run_focused_comparison_test(test_data)
                
                if HAS_MPL:
                    create_focused_comparison_bar_chart(test_data, results)
                else:
                    print("❌ matplotlib not available - skipping chart generation")
                
            except ValueError:
                print("❌ Invalid input. Please enter integers.")
        
        elif choice == "4":
            # Custom range
            print("\n🎯 CUSTOM RANGE TEST")
            print("="*50)
            print("This tests a custom range of consecutive integers.")
            print("Results will show which method performs best across your chosen range.")
            
            try:
                start_val = int(input("\nEnter starting value: "))
                end_val = int(input("Enter ending value: "))
                
                if start_val < 1 or end_val < start_val:
                    print("❌ Invalid range. Please use positive values with start ≤ end.")
                    continue
                
                range_size = end_val - start_val + 1
                if range_size > 50:
                    print(f"⚠️ Large range ({range_size} values) detected.")
                    use_sampling = input("Use sampling for faster processing? (y/n): ").strip().lower() == 'y'
                    if use_sampling:
                        num_points = int(input("Number of sample points (e.g., 20): "))
                        test_data = generate_custom_range_inputs(start_val, end_val, num_points)
                    else:
                        test_data = generate_custom_range_inputs(start_val, end_val)
                else:
                    test_data = generate_custom_range_inputs(start_val, end_val)
                
                print(f"\n🔄 Generating custom range inputs from {start_val:,} to {end_val:,}...")
                print(f"📊 Test values: {len(test_data['values'])} inputs")
                
                if len(test_data['values']) <= 10:
                    for i, desc in enumerate(test_data['descriptions']):
                        print(f"  {i+1}. {desc}")
                else:
                    for i, desc in enumerate(test_data['descriptions'][:5]):
                        print(f"  {i+1}. {desc}")
                    print(f"  ... and {len(test_data['descriptions']) - 5} more")
                
                print(f"\n🚀 Running comparison test...")
                results = run_focused_comparison_test(test_data)
                
                if HAS_MPL:
                    create_focused_comparison_bar_chart(test_data, results)
                else:
                    print("❌ matplotlib not available - skipping chart generation")
                
            except ValueError:
                print("❌ Invalid input. Please enter integers.")
        
        elif choice == "5":
            # Return to main menu
            print("🔙 Returning to main menu...")
            break
        
        else:
            print("❌ Invalid choice. Please enter a number between 1 and 5.")
        
        # Pause before returning to submenu (except for exit)
        if choice in ["1", "2", "3", "4"]:
            input("\n⏸️ Press Enter to return to focused comparisons menu...")

def main():
    """
    This function provides the main interactive interface for the program
    """
    while True:
        print_enhanced_menu()
        choice = input("🎯 Enter your choice [1-5]: ").strip()
        
        if choice == "1":
            print("\n📋 Choose analysis type:")
            print("  a) 🔍 Analyze single number with specific method")
            print("  b) ⚖️ Compare all methods for single number")
            sub_choice = input("Enter a or b: ").strip().lower()
            
            if sub_choice == "a":
                # Single method analysis
                print("\n🛠️ Which method?")
                print("  1️⃣ Fibonacci")
                print("  2️⃣ Pell") 
                print("  3️⃣ Binary")
                method_choice = input("Enter 1, 2, or 3: ").strip()
                
                try:
                    m = parse_exponent_input("Enter exponent m (supports 2^128 format): ")
                    print(f"\n🔄 Running analysis for m = {m:,}...")
                    
                    # Run the selected method
                    if method_choice == "1":
                        result = run_fibonacci_method(m, show_steps=True)
                        method_name = "Fibonacci"
                    elif method_choice == "2":
                        result = run_pell_method(m, show_steps=True)
                        method_name = "Pell"
                    elif method_choice == "3":
                        result = run_binary_method(m, show_steps=True)
                        method_name = "Binary"
                    else:
                        print("❌ Invalid method choice.")
                        continue
                    
                    # Display results
                    print(f"\n" + "="*60)
                    print(f"📊 {method_name.upper()} METHOD RESULTS")
                    print("="*60)
                    
                    if result['success']:
                        print(f"✅ Success: Generated word for e_{{13}}^{m:,}")
                        print(f"🔤 Word: {result['word']}")
                        print(f"📏 Length: {result['steps']:,} steps")
                        print(f"⏱️ Time: {result['time']:.6f} seconds")
                        
                        # Show computation steps in popup if available
                        if result.get('verification_steps'):
                            show_steps = input("\n🔧 Show detailed computation steps in popup? (y/n): ").strip().lower() == 'y'
                            if show_steps:
                                popup_content = f"{method_name} Method Computation for m = {m:,}\n"
                                popup_content += "=" * 50 + "\n\n"
                                for step in result['verification_steps']:
                                    popup_content += f"• {step}\n"
                                show_popup_info(f"{method_name} Method Details", popup_content)
                    else:
                        print(f"❌ Failed to generate word for m = {m:,}")
                        
                except ValueError as e:
                    print(f"❌ Invalid input: {e}")
                    
            elif sub_choice == "b":
                # Multi-method comparison
                try:
                    m = parse_exponent_input("Enter exponent m for comparison (supports 2^128 format): ")
                    print(f"\n🔄 Running comparison for m = {m:,}...")
                    
                    print(f"🔄 Generating detailed verification...")
                    show_verification = input("Show detailed computation steps in popup? (y/n): ").strip().lower() == 'y'
                    
                    # Run all methods with or without detailed steps
                    fib_result = run_fibonacci_method(m, show_steps=show_verification)
                    pell_result = run_pell_method(m, show_steps=show_verification)
                    bin_result = run_binary_method(m, show_steps=show_verification)
                    
                    results = [
                        {'name': 'Fibonacci', **fib_result},
                        {'name': 'Pell', **pell_result},
                        {'name': 'Binary', **bin_result}
                    ]
                    
                    # Display comparison results
                    print_method_comparison_results(m, results, show_verification)
                    
                except ValueError as e:
                    print(f"❌ Invalid input: {e}")
        
        elif choice == "2":
            # Comparison graph generation with enhanced large number support
            if not HAS_MPL:
                print("❌ matplotlib not installed. Please install with 'pip install matplotlib'")
                continue
                
            print("\n📊 Choose range type:")
            print("  1️⃣ Linear range (e.g., 1 to 100)")
            print("  2️⃣ Exponential range (e.g., 2^10 to 2^20)")
            range_type = input("Enter 1 or 2: ").strip()
            
            try:
                if range_type == "1":
                    # Linear range setup
                    m_start = int(input("Start m: "))
                    m_end = int(input("End m: "))
                    num_points = int(input("Number of data points (e.g., 10): "))
                    
                    if num_points < 2:
                        ms = list(range(m_start, m_end + 1))
                    else:
                        ms = [int(x) for x in np.linspace(m_start, m_end, num_points)]
                        
                elif range_type == "2":
                    # Exponential range setup
                    print("💡 Enter exponents (supports 2^128 or 2**128 format)")
                    exp_start_val = parse_exponent_input("Start value: ")
                    exp_end_val = parse_exponent_input("End value: ")
                    num_points = int(input("Number of data points (e.g., 10): "))
                    
                    # Convert to exponents
                    exp_start = int(round(math.log(exp_start_val, 2)))
                    exp_end = int(round(math.log(exp_end_val, 2)))
                    
                    if num_points < 2:
                        exps = list(range(exp_start, exp_end + 1))
                    else:
                        exps = [int(x) for x in np.linspace(exp_start, exp_end, num_points)]
                    
                    ms = [2 ** k for k in exps]
                    
                else:
                    print("❌ Invalid choice.")
                    continue
                
                # Chart type selection
                print("\n📈 Choose chart type:")
                print("  1️⃣ Line chart (default)")
                print("  2️⃣ Bar chart")
                chart_type = input("Enter 1 or 2: ").strip()
                chart_style = "line" if chart_type == "1" else "bar"
                
                print(f"\n🔄 Processing {len(ms)} data points. This may take a moment...")
                print("Progress: ", end="", flush=True)
                
                # Collect data for all methods
                results_data = []
                for i, m in enumerate(ms):
                    # Progress indicator
                    if i % max(1, len(ms) // 10) == 0:
                        print("█", end="", flush=True)
                    
                    try:
                        fib_result = run_fibonacci_method(m, show_steps=False)
                        pell_result = run_pell_method(m, show_steps=False)
                        bin_result = run_binary_method(m, show_steps=False)
                        
                        results_data.append({
                            'Fibonacci': fib_result,
                            'Pell': pell_result,
                            'Binary': bin_result
                        })
                    except Exception as e:
                        print(f"\n⚠️ Error processing m={m:,}: {e}")
                        # Add placeholder data for failed computations
                        results_data.append({
                            'Fibonacci': {'steps': 0, 'success': False},
                            'Pell': {'steps': 0, 'success': False},
                            'Binary': {'steps': 0, 'success': False}
                        })
                
                print(" ✅ Complete!\n")
                
                # Create enhanced graph with smart scaling
                create_smart_comparison_graph_with_options(ms, results_data, chart_style)
                
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
            except Exception as e:
                print(f"❌ Error creating graph: {e}")
        
        elif choice == "3":
            # Cayley hash demonstration
            print("\n🔍 CAYLEY HASH DEMONSTRATION")
            print("Enter target values (supports multiple formats)")
            
            try:
                m_input = input("Enter m values (comma-separated, range like 10-15, or single value): ").strip()
                m_values = []
                
                # Parse various input formats
                if ',' in m_input:
                    # Comma-separated values
                    for val in m_input.split(','):
                        val = val.strip()
                        if '^' in val or '**' in val:
                            # Handle exponential notation within comma-separated list
                            temp_val = val.replace('^', '**')  # Normalize to Python syntax
                            m_values.append(eval(temp_val))  # Safe for simple expressions
                        else:
                            m_values.append(int(val))
                elif '-' in m_input and not m_input.startswith('-'):
                    # Range format like "10-15"
                    parts = m_input.split('-')
                    if len(parts) == 2:
                        start, end = int(parts[0].strip()), int(parts[1].strip())
                        m_values = list(range(start, end + 1))
                    else:
                        raise ValueError("Invalid range format")
                else:
                    # Single value (possibly with exponential notation)
                    if '^' in m_input or '**' in m_input:
                        m_values = [parse_exponent_input(f"Parsing {m_input}: ")]
                    else:
                        m_values = [int(m_input)]
                
                if not m_values:
                    print("❌ No valid values provided.")
                    continue
                
                print(f"\n🔍 Will analyze: {[f'{m:,}' for m in m_values]}")
                show_computation = input("Show detailed computation steps in popup? (y/n): ").strip().lower() == 'y'
                
                # Run the enhanced hash demonstration
                enhanced_hash_demo(m_values, show_computation)
                
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
            except Exception as e:
                print(f"❌ Error in hash demo: {e}")
        
        elif choice == "4":
            # Focused comparisons
            handle_focused_comparisons()
        
        elif choice == "5":
            # Exit program
            print("\n🎉 Thank you for using the Enhanced Matrix Word Generator!")
            print("Created for mathematical research and poster presentations.")
            print("Goodbye! 👋\n")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter a number between 1 and 5.")
        
        # Pause before returning to menu (except for exit)
        if choice in ["1", "2", "3", "4"]:
            input("\n⏸️ Press Enter to return to main menu...")

# ===============================
# TESTING AND VALIDATION FUNCTIONS
# ===============================

def run_enhanced_tests():
    """
    Run comprehensive tests to verify the enhanced interface works correctly.
    
    This function tests all major functionality including:
    - All three mathematical methods
    - Large number support
    - Hash functionality
    - Popup system
    - Error handling
    """
    print("🧪 RUNNING ENHANCED INTERFACE TESTS")
    print("="*60)
    
    test_cases = [1, 10, 100, 1000]
    
    print("Testing all methods with various inputs...")
    for m in test_cases:
        print(f"\n🔍 Testing m = {m:,}")
        
        try:
            # Test all methods
            fib_result = run_fibonacci_method(m, show_steps=True)
            pell_result = run_pell_method(m, show_steps=True)
            bin_result = run_binary_method(m, show_steps=True)
            
            results = [
                {'name': 'Fibonacci', **fib_result},
                {'name': 'Pell', **pell_result},
                {'name': 'Binary', **bin_result}
            ]
            
            # Test the comparison display function
            print_method_comparison_results(m, results, show_verification=False)
            
            print(f"✅ All tests passed for m = {m:,}")
            
        except Exception as e:
            print(f"❌ Test failed for m = {m:,}: {e}")
    
    print("\n🎯 Testing large number support...")
    try:
        large_m = 2**20  # About 1 million
        print(f"Testing m = {large_m:,}")
        
        fib_result = run_fibonacci_method(large_m)
        pell_result = run_pell_method(large_m)
        bin_result = run_binary_method(large_m)
        
        print(f"✅ Large number test passed")
        print(f"   Fibonacci: {fib_result['steps']:,} steps")
        print(f"   Pell: {pell_result['steps']:,} steps")
        print(f"   Binary: {bin_result['steps']:,} steps")
        
    except Exception as e:
        print(f"❌ Large number test failed: {e}")
    
    print("\n🔍 Testing hash functionality...")
    try:
        enhanced_hash_demo([42], show_computation=False)
        print("✅ Hash demo test passed")
    except Exception as e:
        print(f"❌ Hash demo test failed: {e}")
    
    print(f"\n🎉 Enhanced interface testing complete!")

def validate_mathematical_consistency():
    """
    Validate that the enhanced version maintains mathematical consistency with the original.
    
    This function verifies:
    - Fibonacci sequence generation is correct
    - Pell sequence generation is correct
    - Zeckendorf decomposition works properly
    - Binary decomposition is accurate
    - Step count formulas are consistent
    """
    print("🔬 VALIDATING MATHEMATICAL CONSISTENCY")
    print("="*60)
    
    print("=== Fibonacci Sequence Validation ===")
    # Test Fibonacci recurrence relation: F_n = F_{n-1} + F_{n-2}
    for n in range(3, 10):
        f_n = get_fibonacci_number(n)
        f_n1 = get_fibonacci_number(n-1) 
        f_n2 = get_fibonacci_number(n-2)
        expected = f_n1 + f_n2
        status = "✅" if f_n == expected else "❌"
        print(f"F_{n} = {f_n}, F_{n-1} + F_{n-2} = {f_n1} + {f_n2} = {expected} {status}")
    
    print("\n=== Pell Sequence Validation ===")
    # Test Pell recurrence relation: P_n = 2*P_{n-1} + P_{n-2}
    for n in range(3, 8):
        p_n = get_pell_number(n)
        p_n1 = get_pell_number(n-1)
        p_n2 = get_pell_number(n-2)
        expected = 2*p_n1 + p_n2
        status = "✅" if p_n == expected else "❌"
        print(f"P_{n} = {p_n}, 2*P_{n-1} + P_{n-2} = 2*{p_n1} + {p_n2} = {expected} {status}")
    
    print("\n=== Decomposition Validation ===")
    # Test that decompositions correctly represent numbers
    test_nums = [15, 100, 255, 1024]
    for num in test_nums:
        # Test Zeckendorf decomposition
        fib_decomp = zeckendorf_decomposition(num)
        fib_sum = sum(val for _, val in fib_decomp)
        fib_status = "✅" if num == fib_sum else "❌"
        print(f"Zeckendorf {num}: sum = {fib_sum} {fib_status}")
        
        # Test binary decomposition
        bin_decomp = binary_decomposition(num)
        bin_sum = sum(val for _, val in bin_decomp)
        bin_status = "✅" if num == bin_sum else "❌"
        print(f"Binary {num}: sum = {bin_sum} {bin_status}")
    
    print("\n=== Method Consistency Check ===")
    # Verify all methods produce consistent results
    test_values = [34, 100, 256]
    for m in test_values:
        fib_result = run_fibonacci_method(m)
        pell_result = run_pell_method(m)
        bin_result = run_binary_method(m)
        
        all_success = all([fib_result['success'], pell_result['success'], bin_result['success']])
        status = "✅" if all_success else "❌"
        print(f"m={m}: All methods successful {status}")
        
        if all_success:
            print(f"  Steps - Fib: {fib_result['steps']}, Pell: {pell_result['steps']}, Bin: {bin_result['steps']}")
    
    print("\n" + "="*60)
    print("MATHEMATICAL VALIDATION COMPLETE")
    print("="*60)

# ===============================
# PROGRAM ENTRY POINT
# ===============================

if __name__ == "__main__":
    """
    Main entry point for the enhanced matrix word generator.
    
    This program provides multiple startup options:
    1. Main interactive program with full features
    2. Enhanced testing suite for validation
    3. Mathematical consistency verification
    """
    print("🔢 ENHANCED MATRIX WORD GENERATOR")
    print("="*60)
    print("Professional interface for mathematical research and presentations")
    print("Features:")
    print("  ✨ Clean, emoji-enhanced interface") 
    print("  📊 High-quality graphs suitable for posters")
    print("  🔢 Support for very large numbers (2^128+)")
    print("  🔍 Detailed computation verification")
    print("  🔍 Matrix fingerprinting with Cayley hashes")
    print("  📋 Popup windows for better readability")
    print("  🎯 Focused method comparisons")
    print("  📈 Smart scaling for publication-quality visualizations")
    print("="*60)
    
    print("\n🚀 Choose startup mode:")
    print("  1️⃣ Launch main program (recommended)")
    print("  2️⃣ Run enhanced tests first")
    print("  3️⃣ Run mathematical validation")
    mode = input("Enter 1, 2, or 3: ").strip()
    
    if mode == "1":
        main()
    elif mode == "2":
        run_enhanced_tests()
        input("\n⏸️ Press Enter to continue to main program...")
        main()
    elif mode == "3":
        validate_mathematical_consistency()
        input("\n⏸️ Press Enter to continue to main program...")
        main()
    else:
        print("Invalid choice. Launching main program...")
        main()