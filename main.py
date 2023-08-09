import math
from functools import reduce


class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        res = ''
        n1 = len(word1)
        n2 = len(word2)
        m = min(n1, n2)
        i1 = i2 = 0
        for i in range(m):
            res += word1[i] + word2[i]
            i1 += 1
            i2 += 1
        while i1 < n1:
            res += word1[i1]
            i1 += 1
        while i2 < n2:
            res += word2[i2]
            i2 += 1
        return res

    def divide(self, s: str, t: str) -> bool:
        "Check if t divides s means: s = t+t+...+t"
        if len(s) % len(t) != 0:
            return False
        for i in range(0, len(s) - len(t) + 1, len(t)):
            s1 = s[i: i + len(t)]
            if t != s1:
                return False
        return True

    def gcdOfStrings(self, str1: str, str2: str) -> str:
        x = ''
        max_len = 0
        # s1 is longer than s2
        s1 = str1 if len(str2) <= len(str1) else str2
        s2 = str1 if len(str1) < len(str2) else str2
        for i in range(1, len(s2) + 1):
            t = s1[0:i]
            if self.divide(s1, t) and self.divide(s2, t):
                x = t
                max_len = max(max_len, len(x))
        return x

    def kidsWithCandies(self, candies: list, extraCandies: int) -> list:
        maxCandies = max(candies)
        return [c + extraCandies >= maxCandies for c in candies]

    def canPlaceFlower(self, flowerbed: list, i: int) -> bool:
        n = len(flowerbed)
        if i == 0 and n == 1:
            return flowerbed[i] == 0
        if i == 0 and n > 1:
            return flowerbed[i] == 0 and flowerbed[i + 1] == 0
        if i == n - 1 and n > 1:
            return flowerbed[i] == 0 and flowerbed[i - 1] == 0
        return n >= 3 and flowerbed[i] == 0 and flowerbed[i - 1] == 0 and flowerbed[i + 1] == 0

    def canPlaceFlowers(self, flowerbed: list, n: int) -> bool:
        count = 0
        i = 0
        while i < len(flowerbed):
            if self.canPlaceFlower(flowerbed, i):
                count += 1
                i += 2
            else:
                i += 1
        return n <= count

    def reverseVowels1(self, s: str) -> str:
        res = ''
        vowels = set('aeiou')
        t = [l for l in s if l.lower() in vowels]
        t.reverse()
        for i, l in enumerate(s):
            if l.lower() in vowels:
                res += t[j]
                j += 1
            else:
                res += l
        return res

    def reverseVowels(self, s: str) -> str:
        res = [l for l in s]
        vowels = set('aeiou')
        n = len(s)
        low = 0
        high = n - 1
        while low < high:
            while low < n and s[low].lower() not in vowels:
                low += 1
            while high >= 0 and s[high].lower() not in vowels:
                high -= 1
            if low < high:
                print(low, high, res[low], res[high])
                res[low], res[high] = res[high], res[low]
            low += 1
            high -= 1

        return ''.join(res)

    def reverseWords(self, s: str) -> str:
        words = [w.strip() for w in s.split()]
        return ' '.join(reversed(words))

    def productExceptSelf(self, nums: list) -> list:
        prefix = [1] * len(nums)
        postfix = [1] * len(nums)

        m = 1
        for i, x in enumerate(nums):
            m *= x
            prefix[i] = m

        m = 1
        i = len(nums) - 1
        while i >= 0:
            m *= nums[i]
            postfix[i] = m
            i -= 1

        for i in range(len(nums)):
            if i == 0:
                nums[i] = postfix[i + 1]
            elif i == len(nums) - 1:
                nums[i] = prefix[i - 1]
            else:
                nums[i] = prefix[i - 1] * postfix[i + 1]

        return nums

    def increasingTriplet(self, nums: list) -> bool:
        a = b = c = math.inf
        for x in nums:
            if x <= a:
                a = x
            elif x <= b:
                b = x
            elif x <= c:
                c = x
        return a < b < c and c < math.inf

    def compress(self, chars: list) -> int:
        i = 0
        n = len(chars)
        k = 0
        while i < n:
            j = i
            count = 0
            while j < n and chars[j] == chars[i]:
                count += 1
                j += 1

            if count == 1:
                chars[k] = chars[i]
                k += 1
            else:
                chars[k] = chars[i]
                k += 1
                count_str = str(count)
                for l in count_str:
                    chars[k] = l
                    k += 1
            i = j
        return k

    def moveZeroes(self, nums: list) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[k] = nums[i]
                k += 1
        while k < len(nums):
            nums[k] = 0
            k += 1

    def isSubsequence(self, s: str, t: str) -> bool:
        i = 0
        for lt in t:
            if lt == s[i]:
                i += 1
            if i == len(s):
                return True
        return False

    def calcArea(self, height, i, j):
        return min(height[i], height[j]) * abs(i - j)

    def maxArea(self, height: list) -> int:
        max_area = 0
        n = len(height)
        i = 0
        j = n - 1
        while i < j:
            area = self.calcArea(height, i, j)
            max_area = max(max_area, area)
            if height[i] > height[j]:
                j -= 1
            else:
                i += 1

        return max_area

    def pick_two_numbers(self, nums: list, k: int) -> bool:
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] != -1 and nums[j] != -1 and nums[i] + nums[j] == k:
                    nums[i] = -1
                    nums[j] = -1
                    return True
        return False

    def maxOperations(self, nums: list, k: int) -> int:
        n = len(nums)
        operations = 0
        nums.sort()

        # find first index e where number bigger/equals k
        e = 0
        while e < n and nums[e] < k:
            e += 1

        if e >= n:
            e = n - 1

        s = 0
        while s < e:
            if nums[s] + nums[e] == k:
                operations += 1
                s += 1
                e -= 1
            elif nums[s] + nums[e] > k:
                e -= 1
            else:
                s += 1

        return operations

    def calc_avg(self, nums: list, s: int, e: int, k: int) -> float:
        return sum(nums[s:e]) / k

    def findMaxAverage(self, nums: list, k: int) -> float:
        max_avg = -math.inf
        for i in range(len(nums) - k + 1):
            max_avg = max(self.calc_avg(nums, i, i + k, k), max_avg)
            print(i, i + k, self.calc_avg(nums, i, i + k, k))
        return max_avg

    def decodeString(self, s: str) -> str:
        stack = []
        num = 0
        word = ''
        for l in s:
            print(stack)
            if l == '[':
                stack.append(word)
                stack.append(num)
                word = ''
                num = 0
            elif l == ']':
                k = stack.pop(-1)
                w = stack.pop(-1)
                word = w + (k * word)
            elif l.isdigit():
                num = num * 10 + int(l)
            else:
                word += l
        return word

    def can_enter(self, room_num: int, my_keys: set) -> bool:
        if room_num == 0:
            return True
        return room_num in my_keys

    def visit_room(self, rooms: list, my_keys: set, room_num: int, room_visited: list):
        room_visited[room_num] = True
        for k in rooms[room_num]:
            my_keys.add(k)

    def canVisitAllRooms(self, rooms: list) -> bool:
        my_keys = set(rooms[0])
        room_visited = [False] * len(rooms)
        room_num = 0

        stack = [0]

        while len(stack) > 0:
            room_num = stack.pop()
            if not room_visited[room_num] and self.can_enter(room_num, my_keys):
                self.visit_room(rooms, my_keys, room_num, room_visited)
                for k in my_keys:
                    stack.append(k)
        return all(room_visited)

    def count_ones(self, x: int) -> int:
        c = 0
        while x > 0:
            d = x % 2
            if d == 1:
                c += 1
            x //= 2
        return c

    def countBits(self, n: int) -> list:
        res = []
        for i in range(n + 1):
            ones = self.count_ones(i)
            res.append(ones)
        return res

    def singleNumber(self, nums: list) -> int:
        return reduce(lambda a, b: a ^ b, nums)

    def climbStairs(self, n: int) -> int:
        arr = [0] * (n + 1)
        arr[1] = 1
        arr[2] = 2
        for i in range(3, n + 1):
            arr[i] = arr[i - 1] + arr[i - 2]
        return arr[n]

    def minCostClimbingStairs(self, cost: list) -> int:
        n = len(cost)

        res = [0] * (n)

        res[0] = cost[0]
        res[1] = cost[1]

        if n == 0:
            return res[0]

        if n == 1:
            return res[1]

        for i in range(2, n):
            res[i] = min(res[i - 1], res[i - 2]) + cost[i]
        print(res)

        return min(res[n - 1], res[n - 2])

    def rob(self, nums: list) -> int:
        n = len(nums)
        dp = [0] * (n + 1)

        if n == 0:
            return 0
        if n == 1:
            return nums[0]

        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for i in range(2, n):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

        return max(dp[n - 2], dp[n - 1])

    def uniquePaths(self, m: int, n: int) -> int:
        if m == 1 or n == 1:
            return 1

        dp = []
        for i in range(m):
            line = [1] * n
            dp.append(line)

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        for i in range(m):
            print(dp[i])

        return dp[m - 1][n - 1]

    def minPathSum(self, grid: list) -> int:
        m = len(grid)
        n = len(grid[0])

        if m == 1:
            return sum([grid[0][j] for j in range(n)])

        if n == 1:
            return sum([grid[i][0] for i in range(m)])

        dp = []
        for i in range(m):
            line = [0] * n
            dp.append(line)

        # fill first row
        s = 0
        for j in range(n):
            s += grid[0][j]
            dp[0][j] = s

        # fill first column
        s = 0
        for i in range(m):
            s += grid[i][0]
            dp[i][0] = s

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]

        return dp[m - 1][n - 1]

    def uniquePathsWithObstacles(self, obstacleGrid: list) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        if m == 1 and n == 1:
            return 1 - obstacleGrid[0][0]

        dp = []
        for i in range(m):
            line = [0] * n
            dp.append(line)

        # fill row
        found_obstacle = False
        for j in range(n):
            if not found_obstacle and grid[0][j] == 0:
                dp[0][j] = 1
            else:
                found_obstacle = True
                dp[0][j] = 0

        # fill column
        found_obstacle = False
        for i in range(m):
            if not found_obstacle and obstacleGrid[i][0] == 0:
                dp[i][0] = 1
            else:
                found_obstacle = True
                dp[i][0] = 0

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = (dp[i - 1][j] + dp[i][j - 1]) * (1 - obstacleGrid[i][j])

        for i in range(m):
            print(dp[i])

        return dp[m - 1][n - 1]

    def minimumTotal(self, triangle: list) -> int:
        n = len(triangle)
        grid = []
        for i in range(n):
            line = [0] * n
            grid.append(line)

        l = 0
        for j in range(n):
            k = j
            for i in range(n):
                if i + j < n:
                    grid[i][j] = triangle[k][l]
                    k += 1
            l += 1

        dp = []
        for i in range(n):
            line = [0] * n
            dp.append(line)

        s = 0
        for j in range(n):
            s += grid[0][j]
            dp[0][j] = s

        s = 0
        for i in range(n):
            s += grid[i][0]
            dp[i][0] = s

        for i in range(1, n):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]

        for i in range(n):
            print(dp[i])

        min_value = 99999
        for i in range(n):
            for j in range(n):
                if i + j == n:
                    min_value = min(min_value, dp[i][j])

        return min_value

    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = []
        for i in range(n):
            line = [False] * n
            dp.append(line)

        for i in range(n):
            dp[i][i] = True

        for i in range(n - 1):
            dp[i][i + 1] = (s[i] == s[i + 1])

        k = n - 2
        step = 2
        for a in range(n - 1):
            for i in range(k):
                j = i + step
                dp[i][j] = s[i] == s[j] and dp[i + 1][j - 1]
            k -= 1
            step += 1

        for i in range(n):
            print(dp[i])

        max_len = 0
        res = ''
        for i in range(n):
            for j in range(n):
                if i <= j and dp[i][j]:
                    pal_len = j - i + 1
                    if pal_len > max_len:
                        max_len = pal_len
                        res = s[i:j + 1]

        return res

    def addBinary(self, a: str, b: str) -> str:
        res = ''
        i = len(a) - 1
        j = len(b) - 1
        carry = 0

        while i >= 0 and j >= 0:
            y = int(a[i]) + int(b[j]) + carry
            carry = y // 2
            digit = y % 2
            res = str(digit) + res
            i -= 1
            j -= 1

        k = i if len(a) > len(b) else j
        s = a if len(a) > len(b) else b

        while k >= 0:
            y = int(s[k]) + carry
            carry = y // 2
            digit = y % 2
            res = str(digit) + res
            k -= 1

        if carry > 0:
            res = '1' + res

        return res

    def singleNumber(self, nums: list) -> int:
        res = 0

        for b in range(32):
            count_ones = 0
            for num in nums:
                y = (num >> b) & 1
                count_ones += y

            x = count_ones << b
            res = res | x

        return res

    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        res = 0

        shifts = 0
        while left != right:
            left = left >> 1
            right = right >> 1
            shifts += 1

        left <<= shifts

        return left


if __name__ == '__main__':
    sol = Solution()
    arr = [1, 2, 3, 1]
    m = 5
    n = 7
    res = sol.rangeBitwiseAnd(5,7)
    print(res)
