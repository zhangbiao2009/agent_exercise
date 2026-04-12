from typing import List, Optional


def merge_sorted(list1: Optional[List[int]] = None, list2: Optional[List[int]] = None) -> List[int]:
    """
    Merge two sorted lists into a single sorted list without using built-in sort.
    
    Args:
        list1: First sorted list of integers (can be None or empty)
        list2: Second sorted list of integers (can be None or empty)
    
    Returns:
        A new sorted list containing all elements from both input lists
    
    Raises:
        TypeError: If list1 or list2 contains non-integer values
        ValueError: If list1 or list2 is not sorted in non-decreasing order
    
    Examples:
        >>> merge_sorted([1, 3, 5], [2, 4, 6])
        [1, 2, 3, 4, 5, 6]
        >>> merge_sorted([], [1, 2, 3])
        [1, 2, 3]
        >>> merge_sorted([1, 2, 3], None)
        [1, 2, 3]
        >>> merge_sorted([], [])
        []
    """
    # Handle None inputs by converting to empty lists
    if list1 is None:
        list1 = []
    if list2 is None:
        list2 = []
    
    # Validate input types
    def _validate_list(lst: List[int], name: str) -> None:
        for i, item in enumerate(lst):
            if not isinstance(item, int):
                raise TypeError(f"{name} contains non-integer value at index {i}: {item}")
        
        # Check if list is sorted
        for i in range(len(lst) - 1):
            if lst[i] > lst[i + 1]:
                raise ValueError(f"{name} is not sorted in non-decreasing order. "
                               f"Element at index {i} ({lst[i]}) > element at index {i+1} ({lst[i+1]})")
    
    _validate_list(list1, "list1")
    _validate_list(list2, "list2")
    
    # Initialize pointers and result list
    i = 0  # pointer for list1
    j = 0  # pointer for list2
    result = []
    
    # Merge while both lists have elements
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    
    # Append remaining elements from list1 (if any)
    while i < len(list1):
        result.append(list1[i])
        i += 1
    
    # Append remaining elements from list2 (if any)
    while j < len(list2):
        result.append(list2[j])
        j += 1
    
    return result


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ([1, 3, 5], [2, 4, 6]),
        ([], [1, 2, 3]),
        ([1, 2, 3], []),
        ([], []),
        (None, [1, 2, 3]),
        ([1, 2, 3], None),
        ([1, 1, 2], [1, 3, 4]),
        ([1, 5, 10], [2, 3, 4, 6, 7, 8, 9]),
    ]
    
    for list1, list2 in test_cases:
        result = merge_sorted(list1, list2)
        print(f"merge_sorted({list1}, {list2}) = {result}")
    
    # Additional test cases for validation
    print("\nTesting validation:")
    
    # Test non-integer values
    try:
        merge_sorted([1, 2.5, 3], [4, 5, 6])
    except TypeError as e:
        print(f"TypeError caught: {e}")
    
    # Test non-sorted list
    try:
        merge_sorted([1, 3, 2], [4, 5, 6])
    except ValueError as e:
        print(f"ValueError caught: {e}")