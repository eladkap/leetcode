import math


def twoSum_v1(nums: list, k: int) -> list:
    "O(n^2)"
    n = len(nums)
    pairs = []
    for i in range(n):
        for j in range(i, n):
            if nums[i] + nums[j] == k:
                pairs.append(sorted([nums[i], nums[j]]))
    return pairs


def twoSum_v2(nums: list, k: int) -> list:
    "T(n)=O(n), S(n)=O(n)"
    n = len(nums)
    complement_dict = {}
    pairs = []
    for x in nums:
        complement = k - x
        if complement in complement_dict.keys():
            pairs.append([x, complement])
        else:
            complement_dict[x] = k - x
    return pairs


def threeSum_v1(nums: list) -> list:
    "O(n^3)"
    n = len(nums)
    triplets = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if nums[i] + nums[j] + nums[k] == 0:
                    triplets.append(sorted([nums[i], nums[j], nums[k]]))
    return triplets


def threeSum_v2(nums: list) -> list:
    nums.sort()
    n = len(nums)
    triplets = set()

    for i in range(n):
        a = nums[i]
        tmp = nums[:i] + nums[i + 1:]

        low = 0
        high = len(tmp) - 1
        while low < high:
            s = a + tmp[low] + tmp[high]
            if s == 0:
                triplets.add(tuple(sorted([a, tmp[low], tmp[high]])))
                low += 1
                high -= 1
            elif s < 0:
                low += 1
            elif s > 0:
                high -= 1

    return list(list(t) for t in triplets)


def areAnagrams(s: str, t: str) -> bool:
    ds = {}
    dt = {}
    for l in s:
        if l in ds.keys():
            ds[l] += 1
        else:
            ds[l] = 1
    for l in t:
        if l in dt.keys():
            dt[l] += 1
        else:
            dt[l] = 1
    for l in ds:
        if l not in dt:
            return False
        if ds[l] != dt[l]:
            return False
    return True


def groupAnagrams(strs: list) -> list:
    sorted_form_dict = {}  # KEY: sorted form, VALUE: list of anagram words
    for s in strs:
        sorted_form = ''.join(sorted(s))
        if sorted_form in sorted_form_dict.keys():
            sorted_form_dict[sorted_form].append(s)
        else:
            sorted_form_dict[sorted_form] = [s]
    return list(sorted_form_dict.values())


def count_and_say(s: str) -> str:
    res = []
    i = 0
    while i < len(s):
        j = i
        count = 0
        while j < len(s) and s[i] == s[j]:
            count += 1
            j += 1
        res.extend([str(count), s[i]])
        i = j
    return ''.join(res)


def countAndSay(n: int) -> str:
    res = '1'
    for i in range(1, n):
        res = count_and_say(res)
    return res


def sum_square_digits(n: int) -> int:
    res = 0
    while n > 0:
        d = n % 10
        n = n // 10
        res += d ** 2
    return res


def isHappy(n: int) -> bool:
    s = {n}
    while n != 1:
        n = sum_square_digits(n)
        if n in s:
            return False
        s.add(n)
        if n == 1:
            break
    return True


def myAtoi(s: str) -> int:
    res = 0
    sign = 1
    s = s.strip()
    for l in s:
        if l == '-':
            sign = -1
        elif not l.isdigit():
            break
        else:
            digit = int(l)
            res = res * 10 + digit
    res = res * sign
    if res < (-2) ** 31:
        return (-2) ** 31
    if res > 2 ** 31 - 1:
        return 2 ** 31 - 1
    return res


def findMaxConsecutiveOnes(nums: list) -> int:
    maxOnesLen = 0
    countOnes = 0
    for x in nums:
        if x == 1:
            countOnes += 1
        else:
            maxOnesLen = max(maxOnesLen, countOnes)
            countOnes = 0
    return max(maxOnesLen, countOnes)


def countDigits(num: int) -> int:
    c = 0
    while num > 0:
        num = num // 10
        c += 1
    return c


def findNumbers(nums: list) -> int:
    return len(list(filter(lambda num: countDigits(num) % 2 == 0, nums)))


class Solution:
    def merge_arrays(self, arr1: list, arr2: list) -> list:
        n1 = len(arr1)
        n2 = len(arr2)
        arr3 = [0] * (n1 + n2)
        i1 = i2 = i3 = 0
        while i1 < n1 and i2 < n2:
            if arr1[i1] < arr2[i2]:
                arr3[i3] = arr1[i1]
                i1 += 1
                i3 += 1
            else:
                arr3[i3] = arr2[i2]
                i2 += 1
                i3 += 1
        while i1 < n1:
            arr3[i3] = arr1[i1]
            i1 += 1
            i3 += 1
        while i2 < n2:
            arr3[i3] = arr2[i2]
            i2 += 1
            i3 += 1
        return arr3

    def sortedSquares(self, nums: list) -> list:
        "T(n)=O(n), S(n)=O(n)"
        squares1 = []
        squares2 = []
        for num in nums:
            if num < 0:
                squares1.append(num ** 2)
            else:
                squares2.append(num ** 2)
        squares1.reverse()
        squares = self.merge_arrays(squares1, squares2)
        return squares

    def shift_right(self, arr: list, i: int) -> None:
        j = len(arr) - 1
        while j > i:
            arr[j] = arr[j - 1]
            j -= 1

    def duplicateZeros(self, arr: list) -> None:
        i = 0
        while i < len(arr):
            x = arr[i]
            if x == 0:
                self.shift_right(arr, i)
                i += 1  # skip already inserted zero
            i += 1

    def merge(self, nums1: list, m: int, nums2: list, n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i1 = i2 = i3 = 0
        n1 = m
        n2 = n
        nums3 = [0] * (n + m)
        while i1 < n1 and i2 < n2:
            if nums1[i1] < nums2[i2]:
                nums3[i3] = nums1[i1]
                i1 += 1
                i3 += 1
            else:
                nums3[i3] = nums2[i2]
                i2 += 1
                i3 += 1
        while i1 < n1:
            nums3[i3] = nums1[i1]
            i1 += 1
            i3 += 1
        while i2 < n2:
            nums3[i3] = nums2[i2]
            i2 += 1
            i3 += 1

        for i in range(m + n):
            nums1[i] = nums3[i]

    def removeElement(self, nums: list, val: int) -> int:
        low = 0
        for i, x in enumerate(nums):
            if x != val:
                nums[low] = x
                low += 1

        res = low
        for i in range(low, len(nums)):
            nums[i] = '_'

        return res

    def removeDuplicates(self, nums: list) -> int:
        k = 0  # index of next unique value
        i = 0
        n = len(nums)
        while i < n:
            j = i
            while j < n and nums[j] == nums[i]:
                j += 1
            nums[k] = nums[j - 1]
            k += 1
            i = j
        res = k
        while k < n:
            nums[k] = '_'
            k += 1
        return res

    def checkIfExist2(self, arr: list) -> bool:
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                if arr[i] * 2 == arr[j] or arr[j] * 2 == arr[i]:
                    return True
        return False

    def checkIfExist(self, arr: list) -> bool:
        n = len(arr)
        s = set()
        for x in arr:
            print(x, s)
            y = x * 2
            z = x / 2
            if y in s or z in s:
                return True
            s.add(x)
        return False

    def validMountainArray(self, arr: list) -> bool:
        n = len(arr)
        if len(arr) < 3:
            return False

        i = 0
        while i + 1 < n and arr[i] < arr[i + 1]:
            i += 1

        if i >= n - 1 or i == 0:
            return False

        while i + 1 < n and arr[i] > arr[i + 1]:
            i += 1

        return i == n - 1

    def replaceElements(self, arr: list) -> list:
        maxValue = -1
        i = len(arr) - 1
        while i >= 0:
            maxValue = max(maxValue, arr[i])
            arr[i] = maxValue
            i -= 1

        # shift left and put -1 in the last element
        for i in range(len(arr) - 1):
            arr[i] = arr[i + 1]
        arr[-1] = -1
        return arr

    def moveZeroes(self, nums: list) -> None:
        low = 0
        for i, x in enumerate(nums):
            if x != 0:
                nums[low] = x
                low += 1
        while low < len(nums):
            nums[low] = 0
            low += 1

    def sortArrayByParity(self, nums: list) -> list:
        low = 0
        for i, x in enumerate(nums):
            if x % 2 == 0:
                nums[low], nums[i] = nums[i], nums[low]
                low += 1
        return nums

    def heightChecker(self, heights: list) -> int:
        expected = [x for x in heights]
        expected.sort()
        print(heights)
        print(expected)
        c = 0
        for i in range(len(expected)):
            if heights[i] != expected[i]:
                c += 1
        return c

    def thirdMax(self, nums: list) -> int:
        distincts = set(nums)
        max1 = max(distincts)

        if len(distincts) < 3:
            return max1

        distincts.remove(max1)
        max2 = max(distincts)

        distincts.remove(max2)
        max3 = max(distincts)
        return max3

    def findDisappearedNumbers(self, nums: list) -> list:
        missing = []
        for x in nums:
            index = abs(x) - 1
            nums[index] = -1 * abs(x)

        print(nums)
        return missing

    def removeDuplicates2(self, nums: list) -> int:
        # if len(nums) < 3:
        #     return
        res = 0
        i = 0
        k = 0
        n = len(nums)
        while i < n:
            x = nums[i]
            j = i
            count = 0
            while j < n and nums[j] == x:
                count += 1
                j += 1

            c = 1 if count == 1 else 2
            for t in range(c):
                if k + t < n:
                    nums[k + t] = nums[j - 1]

            if count == 1:
                k += 1
            else:
                k += 2

            i = j

        res = k
        while k < n:
            nums[k] = '_'
            k += 1
        return res

    def mapWord(self, s: str) -> str:
        d = {}
        res = ''
        m = 'a'
        for l in s:
            if l not in d.keys():
                d[l] = m
                res += m
                m = chr(ord(m) + 1)
            else:
                res += d[l]
        return res

    def mapSentence(self, sentence: str) -> str:
        d = {}
        res = ''
        m = 'a'
        words = sentence.split()
        for word in words:
            if word not in d.keys():
                d[word] = m
                res += m
                m = chr(ord(m) + 1)
            else:
                res += d[word]
        return res

    def isIsomorphic(self, s: str, t: str) -> bool:
        ms = self.mapWord(s)
        mt = self.mapWord(t)
        return ms == mt

    def wordPattern(self, pattern: str, s: str) -> bool:
        ms = self.mapSentence(s)
        mp = self.mapWord(pattern)
        print(ms)
        print(mp)
        return ms == mp

    def longestConsecutive(self, nums: list) -> int:
        s = set()
        maxCount = 0
        count = 0
        for x in nums:
            s.add(x)

        for x in nums:
            # x is first item in a subsequence
            if x - 1 not in s:
                count = 0
                y = x
                while y in s:
                    y += 1
                    count += 1
            maxCount = max(maxCount, count)

        return maxCount

    def calc_avg(self, nums: list, s: int, e: int, k: int) -> float:
        _sum = 0
        for i in range(s, e):
            _sum += nums[i]
        return _sum / k

    def findMaxAverage(self, nums: list, k: int) -> float:
        if len(nums) == 1:
            return nums[0]
        max_avg = -math.inf
        for i in range(len(nums) - k + 1):
            max_avg = max(self.calc_avg(nums, i, i + k, k), max_avg)
        return max_avg

    def maxVowels(self, s: str, k: int) -> int:
        VOWELS = set('aeiou')
        max_vowels = 0

        for i in range(k):
            if s[i] in VOWELS:
                max_vowels += 1

        count_vowels = max_vowels

        for i in range(k, len(s)):
            if s[i - k] in VOWELS and s[i] not in VOWELS:
                count_vowels -= 1
            if s[i - k] not in VOWELS and s[i] in VOWELS:
                count_vowels += 1
            max_vowels = max(max_vowels, count_vowels)

        return max_vowels

    def calcOnes(self, nums: list) -> int:
        max_count = 0
        c = 0
        for i in range(len(nums)):
            if nums[i] == -1:
                continue
            if nums[i] == 1:
                c += 1
            else:
                max_count = max(max_count, c)
                c = 0
        return max(max_count, c)

    def longestSubarray(self, nums: list) -> int:
        max_ones = 0

        if nums.count(1) == len(nums):
            return len(nums) - 1

        if nums.count(0) == len(nums):
            return 0

        for i in range(len(nums)):
            if nums[i] == 0:
                nums[i] = -1
                count_ones = self.calcOnes(nums)
                max_ones = max(max_ones, count_ones)
                nums[i] = 0

        return max_ones

    def largestAltitude(self, gain: list) -> int:
        tmp = [0]
        s = 0
        for i in range(len(gain)):
            s += gain[i]
            tmp.append(s)

        return max(tmp)

    def isSumPivot(self, nums: list, index: int) -> bool:
        return sum(nums[:index]) == sum(nums[index + 1:])

    def pivotIndex(self, nums: list) -> int:
        s = 0
        left_arr = [0]
        for i in range(len(nums)):
            s += nums[i]
            left_arr.append(s)

        s = 0
        right_arr = [0]
        i = len(nums) - 1
        while i >= 0:
            s += nums[i]
            right_arr.append(s)
            i -= 1

        right_arr.reverse()

        print(left_arr)
        print(right_arr)

        left = 0
        right = len(nums) - 1

        while left < right:
            if left_arr[left] == right_arr[right]:
                if right - left == 1:
                    return left
                left += 1
                right -= 1
            elif left_arr[left] < right_arr[right]:
                left += 1
            else:
                right -= 1
        return -1

    def findDifference(self, nums1: list, nums2: list) -> list:
        return [set(nums1) - set(nums2), set(nums2) - set(nums1)]

    def removeStars(self, s: str) -> str:
        stack = []
        for l in s:
            stack.append(l)

        res = ''

        stars = 0
        while len(stack) > 0:
            l = stack.pop(-1)
            if l != '*':
                if stars == 0:
                    res = l + res
                else:
                    stars -= 1
            else:
                stars += 1

        return res

    def wordBreak(self, s: str, wordDict: list) -> bool:
        "T(n,m) = O(nm)"
        n = len(s)
        word_dict = set(wordDict)
        dp = [False] * n

        i = n - 1
        while i >= 0:
            # word matches the end of s from i position
            is_match = any([s[i:] == word for word in word_dict])
            if is_match:
                dp[i] = True
            else:
                for word in word_dict:
                    k = len(word)
                    if s[i:].startswith(word) and i + k < n and dp[i + k]:
                        dp[i] = True
            i -= 1

        return dp[0]

    def majorityElement(self, nums: list) -> int:
        n = len(nums)
        d = dict()
        for num in nums:
            if num in d.keys():
                d[num] += 1
            else:
                d[num] = 1
        for num in d.keys():
            if d[num] > math.floor(n / 2):
                return num
        return 0

    def reverse_range(self, arr: list, s: int, e: int) -> None:
        while s < e:
            arr[s], arr[e] = arr[e], arr[s]
            s += 1
            e -= 1

    def rotate(self, nums: list, k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n
        nums.reverse()
        self.reverse_range(nums, 0, k - 1)
        self.reverse_range(nums, k, n - 1)

    def longestCommonPrefix(self, strs: list) -> str:
        longest = ''

        if len(strs) == 0:
            return ''

        first_word = strs[0]
        min_len = min([len(s) for s in strs])
        print(min_len)
        for i in range(1, min_len + 1):
            print(first_word[:i])
            if all([first_word.startswith(s[:i]) for s in strs]):
                longest = first_word[:i]

        return longest

    def isPalindrome(self, s: str) -> bool:
        t = [l.lower() for l in s if l.isalnum()]
        return t == t[::-1]

    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)

        _set = set()
        res = 0
        i = 0
        for j in range(n):
            while s[j] in _set:
                _set.remove(s[i])
                i += 1
            _set.add(s[j])
            res = max(res, j - i + 1)

        return res

    def are_dups_in_arr(self, arr: list) -> bool:
        digits = [l for l in arr if l != '.']
        return len(set(digits)) < len(digits)

    def isValidSudoku(self, board: list) -> bool:
        n = len(board)
        # check rows
        for row in range(n):
            arr = [board[row][j] for j in range(n)]
            if self.are_dups_in_arr(arr):
                return False

        # check columns
        for column in range(n):
            arr = [board[i][column] for i in range(n)]
            if self.are_dups_in_arr(arr):
                return False

        # check sub-boxes
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                arr = []
                for i in range(r, r + 3):
                    for j in range(c, c + 3):
                        arr.append(board[i][j])
                if self.are_dups_in_arr(arr):
                    return False

        return True

    def twoSum(self, nums: list, target: int) -> list:
        res = []
        s = set()
        for i, x in enumerate(nums):
            y = target - x
            if y in s:
                j = nums.index(y)
                return [i, j]
            s.add(x)

        return []

    def containsNearbyDuplicate(self, nums: list, k: int) -> bool:
        n = len(nums)
        d = {}
        for i, x in enumerate(nums):
            # duplicate was found
            if x in d.keys():
                dist = abs(d[x] - i)
                if dist <= k:
                    return True
            d[x] = i
        return False

    def searchInsert(self, nums: list, target: int) -> int:
        n = len(nums)
        low = 0
        high = n - 1
        middle = 0
        while low <= high:
            middle = (low + high) // 2
            if nums[middle] == target:
                return middle
            elif nums[middle] < target:
                low = middle + 1
            else:
                high = middle - 1
        if target < nums[0]:
            return 0
        if target > nums[n - 1]:
            return n
        return middle + 1 if target > nums[middle] else middle

    def binary_search(self, arr: list, target: int) -> int:
        n = len(arr)
        low = 0
        high = n - 1
        middle = 0
        while low <= high:
            middle = (low + high) // 2
            if arr[middle] == target:
                return middle
            elif arr[middle] < target:
                low = middle + 1
            else:
                high = middle - 1
        return -1

    def searchMatrix(self, matrix: list, target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])
        low_row = 0
        high_row = m - 1

        while low_row <= high_row:
            middle_row = (low_row + high_row) // 2
            if matrix[middle_row][0] <= target <= matrix[middle_row][n - 1]:
                return self.binary_search(matrix[middle_row], target)
            elif target > matrix[middle_row][n - 1]:
                low_row = middle_row + 1
            else:
                high_row = middle_row - 1

        return False

    def findPeakElement(self, nums: list) -> int:
        n = len(nums)
        low = 0
        high = n - 1
        middle = 0

        if n == 1:
            return 0

        while low <= high:
            middle = (low + high) // 2
            if middle - 1 >= 0 and middle + 1 < n:
                if nums[middle] > nums[middle + 1] and nums[middle] > nums[middle - 1]:
                    return middle
            if middle == 0:
                return 0 if nums[0] > nums[1] else 1

            if middle == n - 1:
                return n - 1 if nums[n - 1] > nums[n - 2] else n - 2

            if nums[middle] < nums[middle + 1]:
                low = middle + 1
            else:
                high = middle - 1
        return -1

    def searchInRotatedSorted(self, nums: list, target: int) -> int:
        n = len(nums)
        low = 0
        high = n - 1
        while low <= high:
            middle = (low + high) // 2
            if nums[middle] == target:
                return middle

            if nums[low] <= nums[middle]:
                if target <= nums[low]:
                    low = middle + 1
                else:
                    high = middle - 1
            else:
                if target >= nums[high]:
                    high = middle - 1
                else:
                    low = middle + 1

        return -1

    def searchRange(self, nums: list, target: int) -> list:
        n = len(nums)
        low = 0
        high = n - 1
        while low <= high:
            middle = (low + high) // 2
            if nums[middle] == target:
                return middle
            elif nums[middle] < target:
                low = middle + 1
            else:
                high = middle - 1
        return [-1, -1]

    def findMin(self, nums: list) -> int:
        n = len(nums)
        low = 0
        high = n - 1

        if n == 1:
            return nums[0]

        while low <= high:
            middle = (low + high) // 2
            if nums[low] <= nums[high]:
                return nums[low]
            if nums[middle] < nums[middle - 1]:
                return nums[middle]
            if nums[middle] >= nums[low]:
                low = middle + 1
            else:
                high = middle - 1
        return -1

    def searchRange(self, nums: list, target: int) -> list:
        n = len(nums)
        low = 0
        high = n - 1

        index = self.binary_search(nums, target)
        if index == -1:
            return [-1, -1]

        if n == 1:
            return [0, 0]

        if n == 2:
            if nums[0] == nums[1]:
                return [0, 0]
            return [0, 1]

        left = 0
        while low <= high:
            m = (low + high) // 2
            if nums[m] == target and nums[m - 1] < target:
                left = m
                break
            elif target > nums[m]:
                low = m + 1
            else:
                high = m - 1

        right = 0
        low = 0
        high = n - 1
        while low <= high:
            m = (low + high) // 2
            if nums[m] == target and nums[m + 1] > target:
                right = m
                break
            elif target > nums[m]:
                low = m + 1
            else:
                high = m - 1

        return [left, right]

    def sortedSquares3(self, nums: list) -> list:
        n = len(nums)
        i = 0
        while i < n:
            if nums[i] >= 0:
                break
            i += 1

        arr2 = nums[:i]
        arr1 = nums[i:]
        arr2.reverse()

        for i in range(len(arr1)):
            arr1[i] **= 2

        for i in range(len(arr2)):
            arr2[i] **= 2

        n1 = len(arr1)
        n2 = len(arr2)
        i = 0
        j = 0
        k = 0
        while i < n1 and j < n2:
            if arr1[i] < arr2[j]:
                nums[k] = arr1[i]
                i += 1
                k += 1
            else:
                nums[k] = arr2[j]
                j += 1
                k += 1
        while i < n1:
            nums[k] = arr1[i]
            i += 1
            k += 1
        while j < n2:
            nums[k] = arr2[j]
            j += 1
            k += 1

        return nums

    def customSortString(self, order: str, s: str) -> str:
        res = ''
        order_dict = {}
        for i, l in enumerate(order):
            order_dict[l] = i

        for i, l in enumerate(s):
            if l not in order_dict.keys():
                order_dict[l] = i

        letters = list(s)
        letters.sort(key=lambda l: order_dict[l])
        res = ''.join(letters)

        return res

    def reverse_bin_num(self, num: int) -> str:
        b = ''
        while num > 0:
            d = num % 2
            b = b + str(d)
            num //= 2

        res = 0
        j = 0
        for i in range(len(b) - 1, -1, -1):
            res = res + int(b[i]) * 2 ** j
            j += 1

        return res

    def sort_012_sol1(self, arr: list):
        d = {k: 0 for k in range(3)}
        for num in arr:
            d[num] += 1

        i = 0
        for k in d.keys():
            while d[k] > 0:
                arr[i] = k
                i += 1
                d[k] -= 1

        return arr

    def sort_012_sol2(self, arr: list):
        l = 0
        m = 0
        r = len(arr) - 1
        while m <= r:
            x = arr[m]
            if x == 0:
                arr[m], arr[l] = arr[l], arr[m]
                l += 1
                m += 1
            if x == 2:
                arr[m], arr[r] = arr[r], arr[m]
                r -= 1
            if x == 1:
                m += 1

    def twoSum2(self, nums: list, target: int) -> list:
        d = {}
        for i, x in enumerate(nums):
            y = target - x
            if y in d.keys():
                j = d[y]
                return [j, i]
            d[x] = i
        return []

    def find_max_distance_between_identical_chars(self, s: str):
        d = {}
        max_distnace = 0
        max_ch = s[0]
        for i, ch in enumerate(s):
            if ch not in d.keys():
                d[ch] = i
            else:
                j = d[ch]
                if abs(j - i) > max_distnace:
                    max_distnace = abs(j - i)
                    max_ch = ch
        return max_distnace, max_ch

    def is_power_of_2(self, x):
        return x > 0 and x & (x - 1) == 0


if __name__ == '__main__':
    sol = Solution()
    s = 'babdacdbaac'
    res = sol.find_max_distance_between_identical_chars(s)
    print(res)
    for x in range(0, 1000):
        if sol.is_power_of_2(x):
            print(x)
