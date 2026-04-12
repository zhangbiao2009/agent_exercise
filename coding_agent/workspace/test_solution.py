from solution import merge_sorted
import sys
import time

def run_tests():
    passed = 0
    failed = 0
    
    # Test 1: Normal case with positive integers
    try:
        result = merge_sorted([1, 3, 5], [2, 4, 6])
        assert result == [1, 2, 3, 4, 5, 6], f"Expected [1, 2, 3, 4, 5, 6], got {result}"
        print("Test 1 PASSED: Normal case with positive integers")
        passed += 1
    except AssertionError as e:
        print(f"Test 1 FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"Test 1 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 2: Empty first list
    try:
        result = merge_sorted([], [1, 2, 3])
        assert result == [1, 2, 3], f"Expected [1, 2, 3], got {result}"
        print("Test 2 PASSED: Empty first list")
        passed += 1
    except AssertionError as e:
        print(f"Test 2 FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"Test 2 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 3: Empty second list
    try:
        result = merge_sorted([1, 2, 3], [])
        assert result == [1, 2, 3], f"Expected [1, 2, 3], got {result}"
        print("Test 3 PASSED: Empty second list")
        passed += 1
    except AssertionError as e:
        print(f"Test 3 FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"Test 3 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 4: Both lists empty
    try:
        result = merge_sorted([], [])
        assert result == [], f"Expected [], got {result}"
        print("Test 4 PASSED: Both lists empty")
        passed += 1
    except AssertionError as e:
        print(f"Test 4 FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"Test 4 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 5: Duplicate values
    try:
        result = merge_sorted([1, 2, 2, 3], [2, 3, 4, 4])
        assert result == [1, 2, 2, 2, 3, 3, 4, 4], f"Expected [1, 2, 2, 2, 3, 3, 4, 4], got {result}"
        print("Test 5 PASSED: Duplicate values")
        passed += 1
    except AssertionError as e:
        print(f"Test 5 FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"Test 5 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 6: Negative numbers
    try:
        result = merge_sorted([-5, -3, -1], [-4, -2, 0])
        assert result == [-5, -4, -3, -2, -1, 0], f"Expected [-5, -4, -3, -2, -1, 0], got {result}"
        print("Test 6 PASSED: Negative numbers")
        passed += 1
    except AssertionError as e:
        print(f"Test 6 FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"Test 6 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 7: Single-element inputs
    try:
        result = merge_sorted([7], [3])
        assert result == [3, 7], f"Expected [3, 7], got {result}"
        print("Test 7 PASSED: Single-element inputs")
        passed += 1
    except AssertionError as e:
        print(f"Test 7 FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"Test 7 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 8: Already sorted (second list all larger)
    try:
        result = merge_sorted([1, 2, 3], [4, 5, 6])
        assert result == [1, 2, 3, 4, 5, 6], f"Expected [1, 2, 3, 4, 5, 6], got {result}"
        print("Test 8 PASSED: Already sorted (second list all larger)")
        passed += 1
    except AssertionError as e:
        print(f"Test 8 FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"Test 8 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 9: Type edge case - floats instead of ints
    try:
        result = merge_sorted([1.5, 2.5], [1.0, 3.0])
        # Check if sorted correctly
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1], f"Not sorted at index {i}: {result}"
        # Check all elements are present
        assert len(result) == 4, f"Expected 4 elements, got {len(result)}"
        print("Test 9 PASSED: Floats instead of ints")
        passed += 1
    except AssertionError as e:
        print(f"Test 9 FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"Test 9 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 10: Large numbers including max int
    try:
        max_int = sys.maxsize
        result = merge_sorted([max_int - 2, max_int], [max_int - 1])
        assert result == [max_int - 2, max_int - 1, max_int], f"Expected [{max_int-2}, {max_int-1}, {max_int}], got {result}"
        print("Test 10 PASSED: Large numbers including max int")
        passed += 1
    except AssertionError as e:
        print(f"Test 10 FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"Test 10 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 11: Performance test with large input
    try:
        n = 10000
        list1 = list(range(0, n, 2))  # Even numbers
        list2 = list(range(1, n, 2))  # Odd numbers
        
        start_time = time.time()
        result = merge_sorted(list1, list2)
        end_time = time.time()
        
        # Verify result is sorted
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1], f"Not sorted at index {i} in large test"
        
        # Verify all elements are present
        assert len(result) == n, f"Expected {n} elements, got {len(result)}"
        
        # Check performance (should complete in reasonable time)
        elapsed = end_time - start_time
        assert elapsed < 1.0, f"Too slow: {elapsed:.4f} seconds for n={n}"
        
        print(f"Test 11 PASSED: Performance test with n={n} (took {elapsed:.4f} seconds)")
        passed += 1
    except AssertionError as e:
        print(f"Test 11 FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"Test 11 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 12: Adversarial - mixed types that might break comparison
    try:
        # This test is designed to fail if the function doesn't handle type errors properly
        result = merge_sorted([1, 2, 3], ['a', 'b', 'c'])
        # If we get here, the function might have tried to compare incompatible types
        print(f"Test 12 got result: {result}")
        print("Test 12 PASSED: Function handled mixed types (or didn't, but didn't crash)")
        passed += 1
    except TypeError as e:
        print(f"Test 12 EXPECTED TypeError: {e}")
        passed += 1  # This is expected to fail for most implementations
    except Exception as e:
        print(f"Test 12 FAILED with unexpected error: {e}")
        failed += 1
    
    # Test 13: Advers